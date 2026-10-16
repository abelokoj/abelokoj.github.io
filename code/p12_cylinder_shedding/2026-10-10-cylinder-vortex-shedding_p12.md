---
layout: post
title: "Drag, Lift and Strouhal Number at Re = 100: A From-Scratch Lattice Boltzmann Solver"
date: 2026-10-10
description: "A D2Q9 lattice Boltzmann solver with interpolated bounce-back, verified on the Schäfer–Turek 2D-2 benchmark and on the unconfined cylinder at Re = 100: drag, lift and Strouhal number under grid refinement."
tags: [CFD, lattice-Boltzmann, vortex-shedding, benchmark, Python, Numba]
giscus_comments: true
related_posts: false
published: false
toc:
  - name: Motivation
  - name: 1. Two test problems
  - name: 2. The lattice Boltzmann method
  - name: 3. Boundaries
  - name: 4. The channel benchmark
  - name: 5. The unconfined cylinder
  - name: 6. Limitations
  - name: Code
  - name: References
---

## Motivation

Place a circular cylinder in a steady stream and, above a Reynolds number near 49, the wake stops being steady: vortices detach alternately from the top and the bottom of the cylinder and form the Kármán vortex street {% cite williamson1996vortex %}. The cylinder feels a drag that oscillates at twice the shedding frequency and a lift that oscillates at the shedding frequency itself, and the dimensionless frequency—the Strouhal number $$\mathrm{St} = fD/U$$—is one of the most carefully measured numbers in fluid mechanics.

This is the second of two Computational Fluid Dynamics (CFD) mini-projects; the first followed the Kelvin–Helmholtz instability of a shear layer with a spectral method ([p10]({% post_url 2026-07-22-kelvin-helmholtz-instability_p10 %})). Here the geometry is the point: a curved, solid boundary, which a spectral method on a periodic box cannot represent. We use the lattice Boltzmann method instead, and the question is the same as before:

> Does a compact lattice Boltzmann solver, written from scratch, reproduce the drag, lift and shedding frequency of established benchmarks at $$\mathrm{Re} = 100$$, and how quickly does it converge as the lattice is refined?

We answer it in two parts. First, we run benchmark 2D-2 of Schäfer and Turek {% cite schafer1996benchmark %}, set up within a priority program of the German Research Foundation (DFG) and since known as the DFG benchmark—a cylinder in a channel, for which finite-element reference values are available to four digits {% cite featflow2d2 %}—at three lattice resolutions and two lattice speeds. Then we remove the channel and compare the unconfined cylinder with the high-resolution simulations of Qu et al. {% cite qu2013quantitative %} and Park, Kwon and Choi {% cite park1998numerical %}.

## 1. Two test problems

**The channel benchmark (2D-2).** A cylinder of diameter $$D = 0.1$$ is centered at $$(0.2, 0.2)$$ in a channel $$[0, 2.2] \times [0, 0.41]$$. The inflow is parabolic with mean velocity $$\bar U = 1$$ (maximum $$1.5$$), the walls are no-slip, and the kinematic viscosity is $$\nu = 10^{-3}$$, so $$\mathrm{Re} = \bar U D/\nu = 100$$. Note that the cylinder sits $$0.005$$ below the channel axis; this small asymmetry is part of the benchmark, it selects the sign of the mean lift, and it is enough to start the shedding without any artificial perturbation. The benchmark quantities are the extreme values of the drag and lift coefficients over a shedding cycle,

$$
C_D = \frac{2 F_D}{\rho \bar U^2 D}, \qquad C_L = \frac{2 F_L}{\rho \bar U^2 D}, \tag{1}
$$

and the Strouhal number $$\mathrm{St} = fD/\bar U$$, where a cycle runs from one minimum of $$C_L$$ to the next {% cite featflow2d2 %}.

**The unconfined cylinder.** The same cylinder in a uniform stream, which is the configuration that experiments and most simulations report. We approximate it with a box 45 diameters long and 40 wide (a blockage of 2.5 percent), with the cylinder 15 diameters from the inlet and free-slip side boundaries. Because this configuration is symmetric, a short transverse velocity bump is placed behind the cylinder at the start to trigger the shedding.

## 2. The lattice Boltzmann method

Why a lattice Boltzmann method? Instead of discretizing the Navier–Stokes equations directly, the method evolves particle distribution functions $$f_i(\mathbf{x}, t)$$ on a regular lattice, each moving with one of a small set of discrete velocities $$\mathbf{c}_i$$; the density and momentum are the moments $$\rho = \sum_i f_i$$ and $$\rho \mathbf{u} = \sum_i \mathbf{c}_i f_i$$ {% cite kruger2017lattice %}. One time step has two parts: a local collision, in which each node relaxes its distributions toward an equilibrium, and streaming, in which each distribution moves to the neighboring node along its velocity,

$$
f_i(\mathbf{x} + \mathbf{c}_i, t + 1) = f_i^{*}(\mathbf{x}, t), \tag{2}
$$

where $$f_i^*$$ is the post-collision value (lattice units, $$\Delta x = \Delta t = 1$$). We use the D2Q9 lattice (nine velocities: rest, four axis neighbors and four diagonals) and the second-order equilibrium

$$
f_i^{\mathrm{eq}} = w_i \rho \left( 1 + 3\,\mathbf{c}_i \cdot \mathbf{u} + \tfrac{9}{2} (\mathbf{c}_i \cdot \mathbf{u})^2 - \tfrac{3}{2} \vert \mathbf{u}\vert ^2 \right), \tag{3}
$$

with weights $$w_0 = 4/9$$, $$w_{1\ldots4} = 1/9$$ and $$w_{5\ldots8} = 1/36$$. In the low-Mach-number limit the scheme recovers the incompressible Navier–Stokes equations with kinematic viscosity $$\nu = (\tau - 1/2)/3$$, where $$\tau$$ is the relaxation time. In practice, the method's appeal for this problem is that all the work is local and uniform: the collision touches one node and the streaming touches its eight neighbors, so the kernels parallelize trivially, and a curved wall costs only a list of boundary links.

**Collision.** The simplest collision (BGK) relaxes all distributions with a single time $$\tau$$. We use the two-relaxation-time (TRT) collision of Ginzburg et al. {% cite ginzburg2008two %} instead, which relaxes the symmetric and antisymmetric parts of each pair $$(f_i, f_{\bar\imath})$$ with separate rates. The symmetric rate sets the viscosity; the antisymmetric one is chosen so that the "magic" combination $$\Lambda = (\tau_+ - \tfrac12)(\tau_- - \tfrac12) = 3/16$$, which places a bounce-back wall exactly halfway between nodes independently of the viscosity. This matters here because some of our runs have $$\tau_+$$ as small as 0.51, where the effective position of a BGK wall shifts noticeably with the viscosity.

**Units.** A lattice simulation fixes the resolution $$N$$ (lattice nodes per diameter) and the inflow speed $$u_0$$ in lattice units; the viscosity then follows from the Reynolds number, $$\nu = u_0 N/\mathrm{Re}$$. The speed $$u_0$$ is a Mach number in disguise: the lattice sound speed is $$1/\sqrt{3}$$, and the compressibility error of the method scales like $$\mathrm{Ma}^2$$. We run every resolution of the channel benchmark at two lattice speeds, $$u_0 = 0.04$$ and $$0.02$$ ($$\mathrm{Ma} \approx 0.07$$ and $$0.035$$ for the mean speed), so that the compressibility error can be measured directly instead of assumed small.

## 3. Boundaries

Most of the effort in a lattice Boltzmann code goes into the boundaries, and this one is no exception.

- **The cylinder.** A node next to the cylinder is missing the distributions that would stream out of the solid. The simplest fix, bounce-back, returns them along the link they arrived on and effectively places the wall halfway along the link—a staircase approximation of the circle. We use the interpolated bounce-back of Bouzidi, Firdaouss and Lallemand {% cite bouzidi2001momentum %}, which uses the exact fraction $$q$$ of each link that lies in the fluid and interpolates linearly between neighboring nodes, so the curved wall is represented to second order.
- **Forces.** The force on the cylinder follows from momentum exchange: every distribution that bounces off the wall transfers momentum $$\mathbf{c}_i (f_i^* + f_{\bar\imath})$$, and summing over the boundary links gives the drag and lift directly, with no stress integration on the surface.
- **Inlet and outlet.** The prescribed inflow velocity and a constant outflow density are imposed with the conditions of Zou and He {% cite zou1997pressure %}. In the channel, the inflow is ramped up smoothly over the first 0.5 time units.
- **Side walls.** Halfway bounce-back for the no-slip channel walls; specular reflection (free slip) for the unconfined box.

The solver is written in Python with the kernels compiled by Numba and parallelized over threads. On an eight-core laptop it updates 50 to 70 million lattice nodes per second, so the finest channel run ($$N = 60$$ at $$u_0 = 0.02$$, 900,000 time steps on 325,000 nodes) takes about 75 minutes.

## 4. The channel benchmark

What does the wake look like once the shedding is established? Figure 1 shows the vorticity in the channel at the end of the finest run ($$N = 60$$, $$u_0 = 0.02$$): the two shear layers that leave the cylinder roll up alternately into vortices of opposite sign, which are carried downstream between the boundary layers of the channel walls. The animation below covers the last eight time units of the same run.

<p align="center">
  <img src="{{ '/assets/img/posts/cyl_st_vorticity_p12.png' | relative_url }}" alt="Vorticity in the DFG 2D-2 channel at Re = 100, showing alternating vortices shed from the cylinder" style="width: 100%; max-width: 980px; height: auto; display: block; margin: 0 auto;">
</p>

*Figure 1: Vorticity $$\omega D/\bar U$$ in the 2D-2 channel at $$\mathrm{Re} = 100$$ ($$N = 60$$ lattice nodes per diameter, $$u_0 = 0.02$$). Red is counterclockwise, blue clockwise; the strips along the top and bottom are the vorticity of the channel walls.*

<p align="center">
  <img src="{{ '/assets/img/posts/cyl_st_p12.gif' | relative_url }}" alt="Animation of the vorticity behind the cylinder in the channel over about two and a half shedding cycles" style="width: 100%; max-width: 900px; height: auto; display: block; margin: 0 auto;">
</p>

All runs start from rest and are integrated to $$t = 30$$ (300 convective times $$D/\bar U$$); the benchmark quantities are measured over the last three full cycles between $$t = 25$$ and $$t = 30$$, as the benchmark prescribes {% cite featflow2d2 %}. By then the flow is periodic to plotting accuracy: in the low-Mach runs the cycle-to-cycle spread of the maximum drag is below $$3 \times 10^{-5}$$ and that of the period below $$10^{-5}$$.

| $$N$$ | $$u_0$$ | $$\max C_D$$ | $$\min C_D$$ | $$\max C_L$$ | $$\min C_L$$ | St |
|---|---|---|---|---|---|---|
| 20 | 0.04 | 3.2757 | 3.2111 | 0.9399 | -0.9722 | 0.3015 |
| 40 | 0.04 | 3.2721 | 3.2010 | 1.0086 | -1.0442 | 0.3018 |
| 20 | 0.02 | 3.2544 | 3.1946 | 0.9289 | -0.9605 | 0.3012 |
| 40 | 0.02 | 3.2479 | 3.1825 | 0.9937 | -1.0284 | 0.3015 |
| 60 | 0.02 | 3.2396 | 3.1743 | 0.9924 | -1.0271 | 0.3018 |
| FeatFlow, level 6 | | 3.2200 | 3.1569 | 0.9859 | -1.0206 | 0.30188 |

*Table 1: Benchmark 2D-2. Extreme values of the drag and lift coefficients over the last three shedding cycles and the Strouhal number, against the finest FeatFlow reference ($$Q_2/P_1^{\mathrm{disc}}$$ elements, level 6, time step 1/200) {% cite featflow2d2 %}.*

Three observations summarize the table and Figure 3.

- **The Strouhal number is right at every resolution.** Every run gives $$\mathrm{St}$$ within 0.25 percent of the reference value, $$0.30188$$, including the coarsest ($$N = 20$$), whose cylinder spans only 20 lattice spacings; the finest run is within 0.03 percent. Note that the reference itself is quoted from a cycle length resolved to the time step of the finite-element run, so differences in the fourth digit are within its uncertainty.
- **The lift is converged by $$N = 40$$.** At $$u_0 = 0.02$$ the maximum lift rises from 0.929 at $$N = 20$$ to 0.994 at $$N = 40$$ and then barely changes (0.992 at $$N = 60$$), within 0.7 percent of the reference 0.9859; the minimum behaves the same way, and the mean lift, $$-0.0174$$ in both low-Mach runs at $$N = 40$$ and $$60$$, matches the reference $$-0.0173$$ to three digits.
- **The drag is limited by compressibility, not by resolution.** At fixed lattice speed, doubling the resolution from $$N = 20$$ to $$N = 40$$ changes the maximum drag by less than 0.01, whereas halving the lattice speed at fixed resolution lowers it by about 0.02 at both resolutions (Figure 3a). In other words, at $$u_0 = 0.04$$ the drag error is dominated by the $$\mathrm{Ma}^2$$ compressibility error of the method, not by the lattice spacing. If we assume that this error scales like $$u_0^2$$, the pair of runs at $$N = 40$$ suggests that about 0.008 of the remaining excess at $$u_0 = 0.02$$ is still compressibility. At $$u_0 = 0.02$$ the maximum drag then falls slowly with resolution, from 3.254 ($$N = 20$$) to 3.248 and 3.240 ($$N = 60$$), so the finest run is 0.6 percent above the reference. We have not identified the source of the rest of the gap; the remaining compressibility error, the constant-density outlet (which differs from the benchmark's do-nothing condition) and the evaluation of the force by momentum exchange on an interpolated curved boundary are all candidates.

<p align="center">
  <img src="{{ '/assets/img/posts/cyl_st_forces_p12.svg' | relative_url }}" alt="Drag and lift coefficients over two shedding cycles at three resolutions, with the FeatFlow reference extrema" style="width: 100%; max-width: 980px; height: auto; display: block; margin: 0 auto;">
</p>

*Figure 2: Drag (a) and lift (b) coefficients over two shedding cycles, aligned at a minimum of the lift, for $$N = 20$$, 40 and 60 at $$u_0 = 0.02$$. Dashed lines: the maximum and minimum of the FeatFlow level-6 reference {% cite featflow2d2 %}. The drag oscillates at twice the shedding frequency, the lift at the shedding frequency.*

<p align="center">
  <img src="{{ '/assets/img/posts/cyl_st_convergence_p12.svg' | relative_url }}" alt="Maximum drag, maximum lift and Strouhal number against lattice resolution for two lattice speeds, with the reference values" style="width: 100%; max-width: 980px; height: auto; display: block; margin: 0 auto;">
</p>

*Figure 3: Maximum drag (a), maximum lift (b) and Strouhal number (c) against the number of lattice nodes per diameter, at lattice speeds $$u_0 = 0.04$$ (blue) and $$0.02$$ (red), with the FeatFlow reference (dashed). Halving $$u_0$$ moves the drag more than doubling $$N$$ does.*

## 5. The unconfined cylinder

Does the same solver reproduce the free wake? Without the channel walls the flow is symmetric, and the wake needs a nudge to start shedding; the transverse bump placed behind the cylinder at $$t = 0$$ provides it, and by $$t \approx 60\, D/U$$ the shedding has settled into a periodic state. Figure 4 shows the result: the classical Kármán street (Figure 4a), a lift that oscillates sinusoidally (Figure 4b), and a lift spectrum with a single dominant peak (Figure 4c).

<p align="center">
  <img src="{{ '/assets/img/posts/cyl_free_p12.svg' | relative_url }}" alt="Unconfined cylinder at Re = 100: vorticity of the vortex street, drag and lift histories, and the lift spectrum" style="width: 100%; max-width: 980px; height: auto; display: block; margin: 0 auto;">
</p>

*Figure 4: The unconfined cylinder at $$\mathrm{Re} = 100$$ ($$N = 20$$, $$u_0 = 0.02$$). (a) Vorticity of the vortex street. (b) Drag and lift over the last 30 convective times. The fast oscillations on the raw drag (light blue) are acoustic waves, not part of the incompressible solution; the dark blue curve is the drag low-pass filtered at four times the shedding frequency. (c) Normalized lift spectrum, with the measured Strouhal number and the value of Qu et al. {% cite qu2013quantitative %}.*

| | St | mean $$C_D$$ | r.m.s. $$C_L$$ |
|---|---|---|---|
| This post, $$N = 20$$, $$u_0 = 0.04$$ | 0.1668 | 1.366 | 0.236 |
| This post, $$N = 20$$, $$u_0 = 0.02$$ | 0.1666 | 1.363 | 0.232 |
| Qu et al. (2013), case D9 | 0.1648 | 1.319 | 0.225 |
| Park, Kwon and Choi (1998) | 0.165 | 1.33 | 0.235 |

*Table 2: The unconfined cylinder at $$\mathrm{Re} = 100$$. The Strouhal number is the peak of the lift spectrum; the mean drag is averaged over whole shedding periods; the lift is low-pass filtered at twice the shedding frequency before its r.m.s. is taken. Reference values from Table 1 of Qu et al. {% cite qu2013quantitative %}, which also lists the values of Park, Kwon and Choi {% cite park1998numerical %}.*

The shedding frequency is reproduced to about one percent: the lift spectrum peaks at $$\mathrm{St} = 0.1666$$, against 0.1648 in Qu et al. and 0.165 in Park, Kwon and Choi (Table 2). The mean drag and the r.m.s. lift are about three percent above the references. Note that this comparison is less controlled than the channel benchmark, for two reasons. First, our box is narrower (40 diameters, i.e., a blockage of 2.5 percent) than the domains of the reference studies, and confinement raises the drag, the lift and the frequency of a cylinder wake; we have not separated this effect from the others. Second, the coarse lattice ($$N = 20$$) that the large box allowed on a laptop is the resolution at which the channel runs still showed a lift deficit of several percent. Halving the lattice speed changes the Strouhal number and the mean drag by less than 0.3 percent (St 0.1668 to 0.1666, mean drag 1.3665 to 1.3630) and the r.m.s. lift by 1.6 percent (0.236 to 0.232), so compressibility is not the main source of the remaining differences.

The drag in this box also carries a visible acoustic signal (Figure 4b): pressure waves, a by-product of the method's weak compressibility, reflect between the inlet, the outlet and the free-slip sides, and their r.m.s. on the drag (about 0.1 to 0.2) is larger than the drag's own oscillation at twice the shedding frequency (about 0.03 to 0.05). The averages in Table 2 are unaffected, since the waves average out over whole shedding periods, but they rule out reporting the extreme values of the drag for this case, and they make the cycle-by-cycle detection used for the channel unreliable; for that reason the Strouhal number here is taken from the spectrum. A finer lattice in a wider box, with non-reflecting boundaries, is the natural next step, and the README lists the corresponding cluster run.

## 6. Limitations

- The flow is two-dimensional. At $$\mathrm{Re} = 100$$ the real wake is two-dimensional too (three-dimensional instability sets in near $$\mathrm{Re} \approx 190$$ {% cite williamson1996vortex %}), so this is a limitation of scope, not of validity.
- The lattice is uniform, which limited the unconfined runs to $$N = 20$$ in a box of 40 diameters. Refinement near the cylinder, the standard remedy, would cut the cost considerably; it would also complicate the code beyond the purpose of a mini-project. The README lists the finer runs to send to a cluster instead.
- The method is weakly compressible. The runs at two lattice speeds measure the compressibility error, but every result still carries a small part of it, and the acoustic waves it produces contaminate the instantaneous drag in the unconfined box.
- The outlet differs from the benchmark's. The 2D-2 benchmark specifies a "do-nothing" outflow; we impose a constant density, which is the natural lattice Boltzmann counterpart but not identical to it.

## Code

The complete code is in the site repository under [`code/p12_cylinder_shedding`](https://github.com/abelokoj/abelokoj.github.io/tree/main/code/p12_cylinder_shedding): `lbm_core.py` (the solver kernels), `cyl_run.py` (one run per call), `cyl_analyze.py` (cycle-by-cycle benchmark quantities) and `plot_cyl.py` (all figures from the saved data). The README lists the commands and how to send the finest runs to a cluster.

## References

The benchmark is that of Schäfer and Turek {% cite schafer1996benchmark %}, with the reference values from the FeatFlow benchmark pages {% cite featflow2d2 %}; the unconfined values come from Qu et al. {% cite qu2013quantitative %} and Park, Kwon and Choi {% cite park1998numerical %}, and Williamson {% cite williamson1996vortex %} reviews the physics of the wake. The lattice Boltzmann method follows Krüger et al. {% cite kruger2017lattice %}, with the TRT collision of Ginzburg et al. {% cite ginzburg2008two %}, the interpolated bounce-back of Bouzidi et al. {% cite bouzidi2001momentum %} and the inlet and outlet conditions of Zou and He {% cite zou1997pressure %}.

{% bibliography --cited --file blog_references %}
