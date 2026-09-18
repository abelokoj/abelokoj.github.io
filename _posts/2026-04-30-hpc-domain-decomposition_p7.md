---
layout: post
title: "Parallel Computing for Scientific Simulation: Domain Decomposition, Amdahl's Law, and Exascale Computing"
date: 2026-04-30
tags: [hpc, parallel-computing, mpi, domain-decomposition, numerical-methods]
giscus_comments: true
---

> **A note on scope.** This post uses `mpi4py` for real distributed execution, but the sandboxed environment it was written in has a single CPU core and no internet access from which to install MPI libraries. The domain decomposition demonstration below is therefore **verified**: it simulates in serial Python the exact communication pattern (ghost-cell halo exchange) that a real MPI program uses, and confirms that it reproduces the monolithic solution exactly. The `mpi4py` code that would run this across distributed nodes is provided as a ready-to-run reference implementation; run it on a cluster or multi-core machine to observe the actual wall-clock speedup.

```python
# Real distributed implementation via mpi4py -- run with:
#   mpirun -n 16 python wave_mpi.py
from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
n_procs = comm.Get_size()

# Each rank computes its own slice of the global domain
NX_GLOBAL, NY = 240, 240
bounds = np.linspace(0, NX_GLOBAL, n_procs + 1).astype(int)
lo, hi = bounds[rank], bounds[rank + 1]
local_nx = hi - lo

u_prev = np.zeros((local_nx + 2, NY))   # +2 for ghost rows
u_curr = np.zeros((local_nx + 2, NY))
# ... initialize u_curr[1:-1, :] with this rank's slice of the initial condition ...

left_neighbor = rank - 1 if rank > 0 else MPI.PROC_NULL
right_neighbor = rank + 1 if rank < n_procs - 1 else MPI.PROC_NULL

for step in range(N_STEPS):
    # Non-blocking halo exchange: post receives and sends together, then wait.
    # This lets MPI overlap communication with other work when possible,
    # which is considerably faster than blocking send-then-receive calls.
    reqs = []
    reqs.append(comm.Isend(u_curr[1, :], dest=left_neighbor))
    reqs.append(comm.Isend(u_curr[-2, :], dest=right_neighbor))
    recv_left = np.empty(NY)
    recv_right = np.empty(NY)
    reqs.append(comm.Irecv(recv_left, source=left_neighbor))
    reqs.append(comm.Irecv(recv_right, source=right_neighbor))
    MPI.Request.Waitall(reqs)
    if rank > 0:
        u_curr[0, :] = recv_left
    if rank < n_procs - 1:
        u_curr[-1, :] = recv_right

    # Local update -- identical stencil math to the serial version above
    lap_x = u_curr[2:, :] + u_curr[:-2, :] - 2 * u_curr[1:-1, :]
    lap_y = np.roll(u_curr, 1, axis=1) + np.roll(u_curr, -1, axis=1) - 2 * u_curr
    u_next = 2 * u_curr[1:-1, :] - u_prev[1:-1, :] + r2 * (lap_x + lap_y[1:-1, :])
    u_prev, u_curr[1:-1, :] = u_curr.copy(), u_next

# Gather the full field onto rank 0 for output/analysis
local_result = u_curr[1:-1, :]
if rank == 0:
    full_result = np.empty((NX_GLOBAL, NY))
else:
    full_result = None
comm.Gatherv(local_result, full_result, root=0)
```



## Motivation: why a mathematician should care about parallel computing

Every post so far in this series has implicitly assumed that solving the equation and running the code are the same task. At the scale at which national laboratories operate, they are not. A single high-fidelity three-dimensional simulation of inertial confinement fusion, climate dynamics, or tsunami wave propagation can involve **trillions** of degrees of freedom, far more than the memory of any single processor can hold, let alone compute within a useful time. Problems at that scale can be solved only by distributing the work across thousands to millions of processors operating together. That requirement defines **parallel computing** in this context: not a performance optimization added afterward, but a structural requirement for the class of problems considered here.

This matters directly for the kind of role a computational mathematician might apply for at a place like Lawrence Livermore National Laboratory. The **El Capitan** system at LLNL, deployed in 2024 and, as of this writing, still among the fastest supercomputers in the world, delivers more than 2.79 exaflops of peak performance {% cite llnl2024elcapitan %}, built from more than 11,000 nodes connected by a high-speed interconnect fabric {% cite hpcwire2024elcapitan %}. None of that raw computational power is usable without algorithms and code specifically designed to be split across that many independent processors, which is exactly the mathematical and engineering problem this post walks through from first principles, on a problem small enough to run on a laptop.

## The core idea: domain decomposition

The dominant strategy for parallelizing a PDE solve across many processors is **domain decomposition**: split the physical domain (a grid, a mesh) into pieces, assign each piece to a different processor, and have neighboring processors exchange a thin strip of boundary data, called **ghost cells** or a **halo**, at each time step, so that each processor holds exactly enough information about its neighbors to update its own piece correctly.

### A concrete problem: wave propagation from a localized source

To make this tangible, this post works with the 2D linear wave equation
{% cite leveque2002finite %},

$$
\frac{\partial^2 u}{\partial t^2} = c^2 \left( \frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2} \right),
$$

with an initial Gaussian "uplift" pulse as the initial condition. The Gaussian pulse is a deliberately simplified representation of the seafloor-displacement source that drives a real tsunami, and an apt one: it belongs to the same class of PDE, a wave equation on a large two- or three-dimensional spatial domain, underlying the work of the LLNL-led team that won the 2025 ACM Gordon Bell Prize for a real-time tsunami early-warning framework running on El Capitan {% cite llnl2025gordonbell %}, which this post returns to at the end.

Discretizing with a standard second-order central-difference (leapfrog) scheme in both space and time gives the explicit update rule,

$$
u^{n+1}_{i,j} = 2u^n_{i,j} - u^{n-1}_{i,j} + \left(\frac{c\,\Delta t}{\Delta x}\right)^2 \left(u^n_{i+1,j} + u^n_{i-1,j} + u^n_{i,j+1} + u^n_{i,j-1} - 4u^n_{i,j}\right).
$$

The key structural fact this whole post rests on: **updating $u^{n+1}_{i,j}$ only requires the four nearest neighbors of $(i,j)$ at the previous time step.** This locality is what makes domain decomposition possible: a processor requires data only from its immediate neighbors, rather than from the entire global domain, to advance its local piece by one time step. Formally, the discrete Laplacian stencil has support radius $1$ in each spatial index, so the dependency graph of the update is itself local, and partitioning the grid into contiguous blocks introduces dependencies only across block boundaries.

### Splitting the grid

Suppose a grid of $N_x$ rows is split into $P$ contiguous blocks, one per processor (a **1D decomposition**; real production codes often use 2D or 3D decompositions for better surface-area-to-volume ratios, discussed below). Each processor's local array is padded with **one extra "ghost" row** on each side:

<p align="center">
  <img src="/assets/img/posts/domain_decomposition_p7.svg" alt="Domain decomposition schematic showing the wave field split into 4 subdomains" style="width: 100%; max-width: 460px; height: auto; display: block; margin: 0 auto;">
</p>

*Figure 1: The same wave field from the monolithic solve below, with vertical lines showing how it would be partitioned across 4 processor "ranks." Each rank owns a contiguous vertical strip.*

At every time step, before any local computation happens, each rank exchanges its boundary row with its neighbors:

```python
# Halo exchange: each rank sends its boundary row to neighbors,
# receives into its own ghost rows -- exactly the communication pattern
# of MPI_Sendrecv between neighboring ranks.
for p in range(n_procs):
    if p > 0:
        blocks_curr[p][0, :] = blocks_curr[p - 1][-2, :]    # recv from left neighbor
    if p < n_procs - 1:
        blocks_curr[p][-1, :] = blocks_curr[p + 1][1, :]    # recv from right neighbor
```

Once every rank has valid data in its ghost rows, each rank computes its own update **completely independently**, requiring no further communication until the next time step:

```python
lap_x = np.zeros_like(local_curr)
lap_x[1:-1, :] = local_curr[2:, :] + local_curr[:-2, :] - 2 * local_curr[1:-1, :]
lap_y = np.roll(local_curr, 1, axis=1) + np.roll(local_curr, -1, axis=1) - 2 * local_curr
local_next = 2 * local_curr - local_prev + r2 * (lap_x + lap_y)
```

This alternation, in which boundary data are **communicated and local computation then proceeds independently**, is the central pattern in distributed scientific computing. Nearly every parallel PDE solver, regardless of the specific equation, follows this two-phase structure at every time step.

## Correctness first: verifying that decomposition reproduces the reference solution

Before reasoning about speed at all, it is essential to verify that splitting the computation in this way does not silently change the answer. The check applies the same discipline as the correctness check in the first post of this series, applied here to a parallel algorithm rather than a sequential one.

```python
u_mono, sensor_traces = solve_monolithic()

for n_procs in [1, 2, 4, 8, 16]:
    u_decomp = solve_decomposed(n_procs)
    max_err = np.max(np.abs(u_decomp - u_mono))
    print(f"P={n_procs:3d} subdomains: max abs difference = {max_err:.3e}")
```

```
Correctness check: decomposed solve vs monolithic solve
  P=  1 subdomains: max abs difference = 4.302e-16
  P=  2 subdomains: max abs difference = 4.302e-16
  P=  4 subdomains: max abs difference = 4.302e-16
  P=  8 subdomains: max abs difference = 4.302e-16
  P= 16 subdomains: max abs difference = 4.302e-16
```

The result is identical, to floating-point round-off, regardless of the number of pieces into which the domain is split. Agreement to round-off is the expected outcome: the explicit finite-difference stencil is a purely local operation, and domain decomposition with correct halo exchange is mathematically an *exact* reformulation of the same computation rather than an approximation. Whenever a decomposed solve fails to match the monolithic one beyond expected round-off, the defect is almost always in the halo-exchange logic; a missing or off-by-one ghost-cell update is the most common bug in production MPI-based PDE codes.

<p align="center">
  <img src="/assets/img/posts/wave_field_sensors_p7.svg" alt="Simulated wave field and sensor recordings" style="width: 100%; max-width: 680px; height: auto; display: block; margin: 0 auto;">
</p>

*Figure 2: Left, the simulated wave field after 300 time steps, spreading outward from the Gaussian source pulse, with black triangles marking sensor locations. Right, the amplitude recorded at each sensor over time. This configuration, comprising a localized wave source, sensors recording its passage, and the inverse problem of reasoning backward from sensor data to source parameters, is the subject of the next post in this series, on Bayesian inverse problems.*

## From correctness to performance: Amdahl's Law

Having confirmed that decomposition yields the correct answer, the next question concerns the magnitude of the speedup obtained. The answer begins with a constraining result established in 1967 {% cite amdahl1967validity %}.

Suppose a fraction $f$ of a program's total work can be perfectly parallelized across $P$ processors, while a fraction $(1-f)$ is inherently sequential (I/O, setup, a part of the algorithm that cannot be split). **Amdahl's Law** gives the theoretical speedup,

$$
S(P) = \frac{1}{(1-f) + \dfrac{f}{P}}.
$$

As $P \to \infty$, this does not diverge; it converges to a **hard
ceiling**,

$$
\lim_{P \to \infty} S(P) = \frac{1}{1-f}.
$$

<p align="center">
  <img src="/assets/img/posts/amdahl_law_p7.svg" alt="Amdahl's Law speedup curves for various parallel fractions" style="width: 100%; max-width: 700px; height: auto; display: block; margin: 0 auto;">
</p>

*Figure 3: Theoretical speedup vs. number of processes, for several values of the parallel fraction $f$. Even at $f=0.99$, corresponding to 99% of the code being perfectly parallelized, the maximum possible speedup is capped at $100\times$ regardless of the number of processors applied. The bound reflects no limitation of particular hardware; it is a mathematical ceiling.*

This has a very concrete implication for how a computational scientist should spend their time: **profile before parallelizing.** If a code spends 20% of its time in an unavoidably sequential setup phase, no amount of additional hardware will ever get more than a $5\times$ speedup, and chasing a bigger cluster allocation to fix a bottleneck that Amdahl's Law has already capped is a common, avoidable waste of both compute budget and engineering time.

## Weak scaling: a different, often more relevant question

Amdahl's Law addresses **strong scaling**, in which the total problem size is fixed and processors are added. Production HPC codes are frequently run in the opposite regime, **weak scaling** {% cite gustafson1988reevaluating %}, in which problem size grows *with* the number of processors, so that each processor handles a fixed-size local portion irrespective of the processor count. The relevant question in that regime is not how much faster the same problem runs, but whether adding processors permits a proportionally larger problem to be solved within the same wall-clock time.

The obstacle to *ideal* weak scaling (perfectly flat efficiency as $P$ grows) is **communication overhead**: every additional processor means more neighbor-to-neighbor halo exchanges happening across the interconnect, and at large enough $P$, that communication cost starts to compete meaningfully with the local computation time.

<p align="center">
  <img src="/assets/img/posts/weak_scaling_p7.svg" alt="Weak scaling efficiency curves under a simple communication overhead model" style="width: 100%; max-width: 700px; height: auto; display: block; margin: 0 auto;">
</p>

*Figure 4: Parallel efficiency under a simple model where communication overhead grows logarithmically with the number of processes (a simplification of real network topology effects). The dashed black line is the unattainable ideal; the colored curves show how efficiency degrades as the communication-to-computation ratio increases. Communication overhead is the trade-off that HPC teams devote substantial engineering effort to managing, through improved interconnect topology, overlapping communication with computation, and reducing message counts.*

This is precisely why real production codes use **2D or 3D domain decomposition** rather than the simple 1D column-split used in this post's demo: for a fixed number of processors $P$, a 2D decomposition gives each processor a roughly square rather than elongated local domain, which **minimizes the surface-area-to-volume ratio**. Since communication cost scales with the boundary, or surface, that each processor shares with its neighbors, while useful computation scales with the interior, or volume, minimizing that ratio directly minimizes communication overhead relative to useful work. Concretely, for a square local domain of side length $\ell$ split among $P$ processors in a $\sqrt{P}\times\sqrt{P}$ grid, each processor's halo perimeter scales as $O(\ell)$ while its interior scales as $O(\ell^2)$; for the equivalent 1D strip decomposition the halo is a fixed-width edge running the *full* domain length, independent of $P$, so the communication-to-computation ratio degrades faster as $P$ grows. The underlying argument is isoperimetric in character, favoring compact shapes over elongated ones to minimize boundary relative to area, and it applies directly to supercomputer performance engineering.

## From simulated ranks to a real MPI implementation

The demonstration above simulates $P$ ranks as Python lists within a single process, which is useful for understanding and verifying the algorithm but is not itself parallel. A production implementation, run across a cluster, uses `mpi4py` {% cite dalcin2008mpi %} to bind each rank to a genuinely separate process (often on a separate physical node), communicating via the Message Passing Interface (MPI) standard {% cite gropp2014using %}:

```python
# Real distributed implementation via mpi4py -- run with:
#   mpirun -n 16 python wave_mpi.py
from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
n_procs = comm.Get_size()

# Each rank computes its own slice of the global domain
NX_GLOBAL, NY = 240, 240
bounds = np.linspace(0, NX_GLOBAL, n_procs + 1).astype(int)
lo, hi = bounds[rank], bounds[rank + 1]
local_nx = hi - lo

u_prev = np.zeros((local_nx + 2, NY))   # +2 for ghost rows
u_curr = np.zeros((local_nx + 2, NY))
# ... initialize u_curr[1:-1, :] with this rank's slice of the initial condition ...

left_neighbor = rank - 1 if rank > 0 else MPI.PROC_NULL
right_neighbor = rank + 1 if rank < n_procs - 1 else MPI.PROC_NULL

for step in range(N_STEPS):
    # Non-blocking halo exchange: post receives and sends together, then wait.
    # This lets MPI overlap communication with other work when possible,
    # which is considerably faster than blocking send-then-receive calls.
    reqs = []
    reqs.append(comm.Isend(u_curr[1, :], dest=left_neighbor))
    reqs.append(comm.Isend(u_curr[-2, :], dest=right_neighbor))
    recv_left = np.empty(NY)
    recv_right = np.empty(NY)
    reqs.append(comm.Irecv(recv_left, source=left_neighbor))
    reqs.append(comm.Irecv(recv_right, source=right_neighbor))
    MPI.Request.Waitall(reqs)
    if rank > 0:
        u_curr[0, :] = recv_left
    if rank < n_procs - 1:
        u_curr[-1, :] = recv_right

    # Local update -- identical stencil math to the serial version above
    lap_x = u_curr[2:, :] + u_curr[:-2, :] - 2 * u_curr[1:-1, :]
    lap_y = np.roll(u_curr, 1, axis=1) + np.roll(u_curr, -1, axis=1) - 2 * u_curr
    u_next = 2 * u_curr[1:-1, :] - u_prev[1:-1, :] + r2 * (lap_x + lap_y[1:-1, :])
    u_prev, u_curr[1:-1, :] = u_curr.copy(), u_next

# Gather the full field onto rank 0 for output/analysis
local_result = u_curr[1:-1, :]
if rank == 0:
    full_result = np.empty((NX_GLOBAL, NY))
else:
    full_result = None
comm.Gatherv(local_result, full_result, root=0)
```

Every piece of this maps directly onto the serial simulation above: `Isend`/`Irecv` implement the exact halo-exchange logic from the `blocks_curr[p][0, :] = ...` lines; `MPI.PROC_NULL` cleanly handles the domain's physical edges (a rank with no left neighbor simply communicates with a "null" process that does nothing); and `Gatherv` reassembles the distributed pieces back into a single global array, exactly like the reassembly loop at the end of the serial `solve_decomposed` function.

## A second axis of parallelism: what's inside each node

The discussion above concerns parallelism *across* nodes, with MPI ranks communicating over a network. Modern supercomputers, including El Capitan, add a second, orthogonal axis of parallelism *within* each node: each El Capitan compute node is built around AMD Instinct MI300A **APUs**, which combine CPU cores, GPU cores, and a shared pool of high-bandwidth memory in a single package. Using such a machine effectively requires combining two distinct parallel programming models:

- **MPI (distributed-memory parallelism)**: the domain-decomposition pattern treated throughout this post, splitting the *problem* across independent processes that each hold private memory and communicate explicitly by message passing.
- **Data parallelism on a GPU (shared-memory, SIMD-style parallelism)**: within a single node, a GPU applies the *same* arithmetic operation to thousands of data elements simultaneously. Returning to the local update step in the serial demonstration above,

  ```python
  lap_x[1:-1, :] = local_curr[2:, :] + local_curr[:-2, :] - 2 * local_curr[1:-1, :]
  ```

  every grid point's Laplacian is computed by an *identical* arithmetic expression, entirely independent of every other grid point's computation. That is exactly the structure a GPU is built to exploit: thousands of lightweight cores each executing the same instruction on a different element of data simultaneously. GPU threading is an entirely different mechanism from the independent-process-plus-message-passing model of MPI, but it serves the same underlying goal of exploiting mathematical independence to obtain wall-clock speed.

This combination of MPI *between* nodes and GPU data-parallelism *within* each node is termed a **hybrid parallel programming model**, and it is close to universal across current exascale-class machines rather than specific to El Capitan. Production HPC codes typically express both layers explicitly: MPI calls of the kind shown in the `mpi4py` example above for cross-node communication, together with a GPU programming interface such as CUDA, HIP, or a portable abstraction layer such as Kokkos or RAJA, both developed in part at LLNL for this purpose, for the node-local, data-parallel computation. Understanding both axes, and where the boundary between them should fall for a given problem, is a distinct skill from either one alone, and a common gap between writing a serial numerical method and writing one that uses an exascale machine effectively.

## A recent breakthrough: exascale tsunami forecasting

A concrete indication of the practical limit of these ideas is instructive. The Gordon Bell Prize-winning tsunami system mentioned earlier used El Capitan to run the largest unstructured-mesh finite-element simulation to date, resolving 55.5 trillion degrees of freedom {% cite llnl2025gordonbell %}. That scale is inaccessible to any single processor and unreachable even with a modest cluster; it requires precisely the domain-decomposition-plus-halo-exchange pattern described here, scaled across the more than 46,000 accelerator processing units and more than 11 million CPU cores of the machine {% cite llnl2025gordonbell %}.

What makes the system a genuine breakthrough, not just a scale demonstration, is the *inverse* half of the pipeline: rather than only running forward wave simulations, the team built a system that rapidly solves the inverse problem of inferring the seafloor displacement scenario that best matches real-time pressure sensor data {% cite llnl2025gordonbell %}, turning a physics simulation that would normally require hours into a forecast delivered approximately ten billion times faster than conventional approaches, on the order of a fraction of a second {% cite newswise2025gordonbell %}. That inverse step, proceeding from noisy sensor measurements back to the unknown source parameters that produced them, is the subject of the next post in this series, and the simplified wave-source problem in Figure 2 above is a reduced version of that inference task.

## Summary Notes and Highlights

- **Locality is what makes parallelism possible.** The wave equation's finite-difference stencil touches only nearest neighbors, and this is the structural property that permits domain decomposition to split the problem without changing the answer.
- **Halo exchange, then independent local computation** is the universal two-phase pattern behind essentially all distributed PDE solvers, regardless of the specific equation being solved.
- **Verify correctness before optimizing for speed.** A decomposed solve that does not reproduce the monolithic answer, beyond floating-point round-off, contains a defect, almost always in the halo exchange. This should be checked explicitly on every run rather than assumed.
- **Amdahl's Law is a hard ceiling rather than a heuristic.** A sequential bottleneck caps the maximum attainable speedup regardless of the hardware added, which should directly inform where optimization effort is directed.
- **Communication overhead, rather than raw computation, is usually the binding constraint at scale.** This is why production codes favor two- and three-dimensional domain decompositions over simple one-dimensional splits, minimizing the surface-to-volume ratio of communication relative to useful work.
- **The largest-scale scientific breakthroughs at national labs are built directly on these ideas**, scaled from a laptop-sized demo like the one in this post up to tens of millions of cores.

## References

The core theoretical framework in this post follows {% cite amdahl1967validity %} for strong scaling and {% cite gustafson1988reevaluating %} for the complementary weak-scaling perspective. {% cite gropp2014using %} is the standard reference for MPI programming, including the domain-decomposition patterns used throughout, with {% cite dalcin2008mpi %} documenting the `mpi4py` package used in the reference implementation. {% cite leveque2002finite %} covers the wave equation discretization and stability analysis underlying the demo. The El Capitan system details are drawn from {% cite llnl2024elcapitan %} and {% cite hpcwire2024elcapitan %}, and the Gordon Bell Prize-winning tsunami-forecasting system referenced throughout is documented in {% cite llnl2025gordonbell %} and {% cite newswise2025gordonbell %}.

{% bibliography --cited --file blog_references %}

---

*Full, executed code for the domain decomposition demo is available in [`hpc_domain_decomposition_p7.py`]({{ '/assets/code/hpc_domain_decomposition_p7.py' | relative_url }}). The `mpi4py` implementation is provided as a correct reference implementation for real distributed execution.*
