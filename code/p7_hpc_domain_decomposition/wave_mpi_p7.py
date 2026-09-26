"""
wave_mpi_p7.py
==============
Distributed-memory solve of the 2D linear wave equation with mpi4py, using a
1D row decomposition with one ghost row on each side of every subdomain.

It solves the same problem as hpc_domain_decomposition_p7.py (same Gaussian
pulse, grid, CFL number and Dirichlet boundaries), so the two can be checked
against each other. The stencil is evaluated in the same arithmetic order as
the serial monolithic solver, which is why the MPI result can agree with it
exactly rather than merely to round-off.

Usage
-----
    # correctness: gather the field and compare it with the serial solver
    mpirun -n 4 python3 wave_mpi_p7.py --check

    # timing: fixed global grid (strong scaling)
    mpirun -n 8 python3 wave_mpi_p7.py --nx 2048 --ny 2048 --steps 200

    # timing: fixed rows per rank (weak scaling)
    mpirun -n 8 python3 wave_mpi_p7.py --rows-per-rank 256 --ny 2048 --steps 200

Rank 0 prints one line of JSON with the timings, so a driver script can
collect results across runs.
"""
import argparse
import json
import sys

import numpy as np
from mpi4py import MPI


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--nx", type=int, default=240, help="global rows")
    p.add_argument("--ny", type=int, default=240, help="global columns")
    p.add_argument("--rows-per-rank", type=int, default=None,
                   help="weak scaling: set nx = rows_per_rank * n_ranks")
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--cfl", type=float, default=0.5)
    p.add_argument("--check", action="store_true",
                   help="compare the gathered field with a serial solve on rank 0")
    return p.parse_args()


def gaussian_pulse(x, y, amplitude=1.0, x0=0.3, y0=0.5, width=0.03):
    X, Y = np.meshgrid(x, y, indexing="ij")
    return amplitude * np.exp(-((X - x0) ** 2 + (Y - y0) ** 2) / (2 * width ** 2))


def serial_reference(nx, ny, steps, r2):
    """The monolithic solver from hpc_domain_decomposition_p7.py, unchanged."""
    u_prev = gaussian_pulse(np.linspace(0, 1, nx), np.linspace(0, 1, ny))
    u_curr = u_prev.copy()
    for _ in range(steps):
        lap = (np.roll(u_curr, 1, axis=0) + np.roll(u_curr, -1, axis=0)
               + np.roll(u_curr, 1, axis=1) + np.roll(u_curr, -1, axis=1)
               - 4 * u_curr)
        u_next = 2 * u_curr - u_prev + r2 * lap
        u_next[0, :] = u_next[-1, :] = u_next[:, 0] = u_next[:, -1] = 0.0
        u_prev, u_curr = u_curr, u_next
    return u_curr


def main():
    args = parse_args()
    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()

    nx = args.rows_per_rank * size if args.rows_per_rank else args.nx
    ny = args.ny
    if nx < size:
        if rank == 0:
            print("more ranks than rows", file=sys.stderr)
        comm.Abort(1)

    dx = 1.0 / nx          # grid spacing as defined in the serial script
    dt = args.cfl * dx     # c = 1
    r2 = (dt / dx) ** 2

    # ---- this rank's rows [lo, hi) of the global grid ----
    bounds = np.linspace(0, nx, size + 1).astype(int)
    lo, hi = int(bounds[rank]), int(bounds[rank + 1])
    n_local = hi - lo

    x = np.linspace(0, 1, nx)[lo:hi]
    y = np.linspace(0, 1, ny)
    u_prev = np.zeros((n_local + 2, ny))   # rows 0 and -1 are ghost rows
    u_curr = np.zeros((n_local + 2, ny))
    u_prev[1:-1] = gaussian_pulse(x, y)
    u_curr[1:-1] = u_prev[1:-1]            # zero initial velocity

    up = rank - 1 if rank > 0 else MPI.PROC_NULL
    down = rank + 1 if rank < size - 1 else MPI.PROC_NULL
    first_owned_is_boundary = (lo == 0)
    last_owned_is_boundary = (hi == nx)

    t_comm = t_comp = 0.0
    comm.Barrier()
    t_start = MPI.Wtime()

    for _ in range(args.steps):
        # ---- halo exchange: send owned boundary rows, receive ghost rows ----
        t0 = MPI.Wtime()
        comm.Sendrecv(u_curr[1], dest=up, recvbuf=u_curr[-1], source=down)
        comm.Sendrecv(u_curr[-2], dest=down, recvbuf=u_curr[0], source=up)
        t1 = MPI.Wtime()

        # ---- local update, same arithmetic order as the serial solver ----
        c = u_curr[1:-1]
        lap = (u_curr[:-2] + u_curr[2:]
               + np.roll(c, 1, axis=1) + np.roll(c, -1, axis=1)
               - 4 * c)
        u_next = 2 * c - u_prev[1:-1] + r2 * lap
        u_next[:, 0] = u_next[:, -1] = 0.0
        if first_owned_is_boundary:
            u_next[0] = 0.0
        if last_owned_is_boundary:
            u_next[-1] = 0.0
        u_prev[1:-1] = c
        u_curr[1:-1] = u_next
        t2 = MPI.Wtime()

        t_comm += t1 - t0
        t_comp += t2 - t1

    comm.Barrier()
    t_total = MPI.Wtime() - t_start

    # The slowest rank sets the wall-clock time.
    t_total = comm.reduce(t_total, op=MPI.MAX, root=0)
    t_comm_max = comm.reduce(t_comm, op=MPI.MAX, root=0)
    t_comp_max = comm.reduce(t_comp, op=MPI.MAX, root=0)

    result = None
    if args.check:
        counts = comm.gather(n_local * ny, root=0)
        full = np.empty((nx, ny)) if rank == 0 else None
        if rank == 0:
            displs = [0] + [int(v) for v in np.cumsum(counts[:-1])]
            recv = [full, counts, displs, MPI.DOUBLE]
        else:
            recv = None
        comm.Gatherv(np.ascontiguousarray(u_curr[1:-1]), recv, root=0)
        if rank == 0:
            ref = serial_reference(nx, ny, args.steps, r2)
            result = float(np.max(np.abs(full - ref)))

    if rank == 0:
        print(json.dumps({
            "ranks": size, "nx": nx, "ny": ny, "steps": args.steps,
            "wall_s": t_total, "comm_s_max": t_comm_max, "comp_s_max": t_comp_max,
            "max_abs_diff_vs_serial": result,
            "mpi": "%s %s" % (MPI.get_vendor()[0], ".".join(map(str, MPI.get_vendor()[1]))),
        }))


if __name__ == "__main__":
    main()
