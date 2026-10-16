---
layout: post
title: "Kelvin–Helmholtz Instability from Linear Theory to Vortex Pairing"
date: 2026-07-22
description: "A temporally evolving shear layer computed with a pseudo-spectral Navier–Stokes solver, verified against the Rayleigh and Orr–Sommerfeld dispersion relations, then followed through roll-up and vortex pairing."
tags: [CFD, hydrodynamic-stability, Kelvin-Helmholtz, spectral-methods, Python]
giscus_comments: true
related_posts: false
published: true
toc:
  - name: Motivation
  - name: 1. The problem
  - name: 2. Linear stability theory
  - name: 3. The simulation
  - name: 4. The linear stage, verified
  - name: 5. Roll-up and vortex pairing
  - name: 6. Limitations
  - name: Code
  - name: References
---

## Motivation

Two layers of fluid sliding past one another—the wind over the sea, the edge of a jet, the boundary between two atmospheric layers—form one of the simplest flows in fluid mechanics, and one of the most unstable. A small ripple on the interface grows exponentially, rolls up into a row of vortices, and the vortices then merge in pairs, so that the layer thickens in discrete steps. This sequence is the Kelvin–Helmholtz instability, and its first stage is a rare case in which a classical linear theory makes a sharp, testable prediction: the growth rate of every wavelength.

{% assign cylinder_post = site.posts | where: "slug", "cylinder-vortex-shedding_p12" | first %}
This post is the first of two Computational Fluid Dynamics (CFD) mini-projects; the second treats vortex shedding behind a circular cylinder{% if cylinder_post %} ([p12]({{ cylinder_post.url | relative_url }})){% else %} (forthcoming){% endif %}. Both are built around the same discipline: before a simulation is used to show anything, it must reproduce a number that is known independently. The question this post asks is therefore narrow:

> Does a pseudo-spectral Navier–Stokes solver reproduce the growth rates predicted by linear stability theory, and what does the same solver then show once the instability is no longer small?

We answer it in three steps. First, we solve the linear stability problem for the hyperbolic-tangent shear layer with Chebyshev collocation, in both its inviscid (Rayleigh) and viscous (Orr–Sommerfeld) forms, and recover the classical result of Michalke {% cite michalke1964inviscid %}. Next, we write a doubly periodic pseudo-spectral solver for the two-dimensional Navier–Stokes equations and measure the growth rate of single eigenmodes at eight wavenumbers. Finally, we let the instability run into its nonlinear stage and follow the roll-up of the layer into vortices and their pairing, the mechanism that Winant and Browand {% cite winant1974vortex %} identified as the source of mixing-layer growth.

## 1. The problem

We consider a temporally evolving shear layer, i.e., a layer that is periodic in the streamwise direction and grows in time instead of in space. All lengths are scaled by the half-thickness $$\delta$$ of the layer and all velocities by $$U_0$$, half the velocity difference across it, so the base flow is

$$
U(y) = \tanh y, \tag{1}
$$

and the only parameter is the Reynolds number $$\mathrm{Re} = U_0 \delta / \nu$$, which we fix at $$\mathrm{Re} = 1000$$. In two dimensions the incompressible Navier–Stokes equations reduce to a single transport equation for the vorticity $$\omega$$,

$$
\frac{\partial \omega}{\partial t} + u \frac{\partial \omega}{\partial x} + v \frac{\partial \omega}{\partial y} = \frac{1}{\mathrm{Re}} \left( \frac{\partial^2 \omega}{\partial x^2} + \frac{\partial^2 \omega}{\partial y^2} \right), \tag{2}
$$

together with a Poisson equation for the streamfunction $$\psi$$, from which the velocity follows:

$$
\frac{\partial^2 \psi}{\partial x^2} + \frac{\partial^2 \psi}{\partial y^2} = -\omega, \qquad u = \frac{\partial \psi}{\partial y}, \qquad v = -\frac{\partial \psi}{\partial x}. \tag{3}
$$

Note that the pressure has disappeared entirely; this is the main attraction of the vorticity–streamfunction form in two dimensions, since incompressibility is satisfied by construction and no pressure–velocity coupling has to be resolved.

## 2. Linear stability theory

What does the theory predict? We superpose on Eq. (1) a small perturbation with streamfunction $$\psi' = \phi(y)\, e^{i\alpha(x - ct)}$$, where $$\alpha$$ is the streamwise wavenumber and $$c = c_r + i c_i$$ the complex phase speed. Linearizing Eqs. (2) and (3) about $$U(y)$$ gives the Orr–Sommerfeld equation,

$$
(U - c)\left(\phi'' - \alpha^2 \phi\right) - U'' \phi = \frac{1}{i\alpha \mathrm{Re}} \left( \phi'''' - 2\alpha^2 \phi'' + \alpha^4 \phi \right), \tag{4}
$$

with $$\phi \to 0$$ as $$\vert y\vert  \to \infty$$. The perturbation grows like $$e^{\omega_i t}$$ with growth rate $$\omega_i = \alpha c_i$$. Letting $$\mathrm{Re} \to \infty$$ removes the right-hand side and leaves the Rayleigh equation, a second-order problem that needs only $$\phi \to 0$$ at both ends. Because Eq. (1) is antisymmetric, the unstable mode travels with the mean velocity ($$c_r = 0$$) and grows in place.

We discretize both equations with Chebyshev collocation {% cite trefethen2000spectral %} and solve the resulting generalized eigenvalue problems $$A\phi = cB\phi$$ with dense linear algebra. The two problems need different treatments of the infinite domain:

- **Rayleigh.** The real line is mapped onto $$(-1, 1)$$ by $$y = a\xi/\sqrt{1 - \xi^2}$$ with $$a = 4$$, so no truncation is needed. With 200 Chebyshev points the growth rate is converged to eight digits (it changes by less than $$10^{-10}$$ between 160 and 240 points, and by less than $$10^{-8}$$ when $$a$$ ranges from 3 to 6).
- **Orr–Sommerfeld.** The fourth-order operator needs two conditions at each end, $$\phi = \phi' = 0$$, which we impose at $$y = \pm 16$$ with Trefethen's clamped fourth-derivative construction {% cite trefethen2000spectral %} and 240 points. Note that the truncation is harmless here: the eigenfunction decays like $$e^{-\alpha \vert y\vert }$$, and at the most unstable wavenumber $$e^{-16\alpha} \approx 10^{-3}$$.

The inviscid problem gives a maximum growth rate of $$\omega_i = 0.18970$$ at $$\alpha = 0.4449$$, in agreement with the values $$0.1897$$ and $$0.4446$$ reported by Michalke {% cite michalke1964inviscid %} (the small difference in $$\alpha$$ reflects the flatness of the curve near its maximum, where $$\omega_i$$ changes by less than $$10^{-6}$$ over that interval). Viscosity lowers the curve slightly: at $$\mathrm{Re} = 1000$$ the maximum is $$\omega_i = 0.18809$$ at $$\alpha = 0.4428$$, a reduction of less than one percent (Figure 1a). The most unstable wavelength, $$\lambda = 2\pi/\alpha \approx 14.2$$ half-thicknesses, sets the spacing of the vortices that form when the layer rolls up.

## 3. The simulation

The solver integrates Eqs. (2) and (3) on a doubly periodic box with a Fourier pseudo-spectral method: derivatives are computed exactly in wavenumber space, products are formed in physical space, and the Poisson equation (3) becomes the division $$\hat\psi = \hat\omega / k^2$$. Three standard choices complete it:

- **Dealiasing.** Nonlinear products are truncated with the 2/3 rule of Orszag {% cite orszag1971elimination %}, which removes the aliasing errors of the quadratic advection term.
- **Time stepping.** The classical fourth-order Runge–Kutta scheme is applied with an integrating factor $$e^{-\nu k^2 t}$$, so the viscous term is integrated exactly and the time step is limited only by advection. We use $$\Delta t = 0.02$$ (a Courant number of about 0.5).
- **Passive scalar.** A scalar $$c$$ with unit Schmidt number is advected alongside the vorticity, starting from $$c = (1 + U)/2$$, i.e., marking the fluid that began on the fast side of the layer. It has no effect on the flow; it simply makes the roll-up visible.

A single shear layer is not periodic in $$y$$, so the box holds two layers of opposite sign, $$U = \tanh(y - L_y/4) - \tanh(y - 3L_y/4) - 1$$. We keep the layers far apart (a quarter of the box height, at least 14 half-thicknesses, separates each layer from the midplane) and report diagnostics for the lower layer only. Note that the upper layer is the mirror image of the lower one and has the same eigenvalues, so the two do not compete.

The solver is about 250 lines of Python (NumPy and SciPy's multithreaded fast Fourier transforms) and is listed in [Code](#code).

## 4. The linear stage, verified

Does the solver reproduce Figure 1a? To test it as directly as possible, we start each run from the Orr–Sommerfeld eigenmode itself, computed at the wavenumber of the box ($$L_x = 2\pi/\alpha$$, box height $$L_y = 64$$, $$32 \times 512$$ points), at an amplitude of $$10^{-6}$$, so that nonlinear terms are negligible for the whole run. Two details make the comparison exact in principle. First, the base profile slowly spreads by viscous diffusion, whereas Eq. (4) assumes it is frozen; we therefore add the body force $$-\nu \nabla^2 \omega_0$$ that holds the base vorticity $$\omega_0$$ steady (only in this test). Second, the growth rate is measured as the slope of $$\log A(t)$$, where $$A$$ is the $$L^2$$ amplitude of the first streamwise Fourier mode of the vorticity, fitted over $$10 \le t \le 40$$.

| $$\alpha$$ | Orr–Sommerfeld $$\omega_i$$ | Simulation $$\omega_i$$ | Relative difference |
|---|---|---|---|
| 0.150 | 0.113436 | 0.114197 | $$6.7 \times 10^{-3}$$ |
| 0.250 | 0.157679 | 0.157768 | $$5.7 \times 10^{-4}$$ |
| 0.350 | 0.181405 | 0.181412 | $$4.1 \times 10^{-5}$$ |
| 0.4428 | 0.188087 | 0.188087 | $$3.0 \times 10^{-6}$$ |
| 0.550 | 0.179914 | 0.179915 | $$2.2 \times 10^{-6}$$ |
| 0.650 | 0.158650 | 0.158656 | $$4.1 \times 10^{-5}$$ |
| 0.750 | 0.125413 | 0.125483 | $$5.6 \times 10^{-4}$$ |
| 0.850 | 0.080695 | 0.081236 | $$6.7 \times 10^{-3}$$ |

Near the maximum the simulation and the theory agree to six digits. The agreement degrades symmetrically toward both ends of the band, to about 0.7 percent at $$\alpha = 0.15$$ and $$0.85$$. We have not traced this residual in detail; it is consistent with the slower-growing modes being contaminated, over the fitting window, by the small part of the initial condition that is not exactly the eigenmode (the interpolated, truncated eigenfunction excites other modes weakly, and these matter more when the target mode grows more slowly). At the most unstable wavenumber the measured rate is independent of the discretization: it is $$0.188087$$ for every combination of $$n_y = 512$$ or $$1024$$ points across the box and $$\Delta t = 0.1$$ or $$0.05$$, and $$0.188121$$ on the coarsest grid ($$n_y = 256$$).

<p align="center">
  <img src="{{ '/assets/img/posts/kh_dispersion_p10.svg' | relative_url }}" alt="Growth rate against wavenumber for the tanh shear layer: inviscid Rayleigh curve, viscous Orr–Sommerfeld curve at Re = 1000, and growth rates measured from the simulation" style="width: 100%; max-width: 900px; height: auto; display: block; margin: 0 auto;">
</p>

*Figure 1: (a) Temporal growth rate of the tanh shear layer against wavenumber: the inviscid Rayleigh curve (black, with its maximum marked), the Orr–Sommerfeld curve at $$\mathrm{Re} = 1000$$ (blue) and the rates measured from the simulation (red circles). (b) Amplitude of the first Fourier mode in three of the runs, normalized by its initial value, against the theoretical growth $$e^{\omega_i t}$$ (dashed); the curves overlap to plotting accuracy.*

## 5. Roll-up and vortex pairing

What happens once the perturbation is no longer small? For the nonlinear run we double the box, $$L_x = 2 \times 2\pi/0.4428 = 28.38$$, so that it holds two wavelengths of the most unstable mode, and seed both that fundamental (at amplitude $$10^{-3}$$) and its subharmonic, the mode of wavelength $$L_x$$ (at $$10^{-4}$$), each in the shape of its own eigenmode. The box height is $$L_y = 2L_x$$, the grid $$512 \times 1024$$ ($$\Delta x = 0.055$$), and the run is unforced, so the mean profile is free to spread. The run to $$t = 200$$ takes 10,000 time steps and 33 minutes on an eight-core laptop.

Figure 2 shows the passive scalar at six instants, and the sequence is the textbook one {% cite ho1984perturbed %}. At $$t = 20$$ the interface is still nearly flat; the perturbation has grown by a factor of 39 but remains small. By $$t = 40$$ the fundamental has saturated and the layer has rolled up into two vortices, each winding the fluid from both sides into a spiral. Between $$t = 40$$ and $$t = 60$$ the two vortices are nearly steady, rotating and slowly entraining fluid. Meanwhile the subharmonic, which grows more slowly ($$\omega_i = 0.147$$), displaces one vortex upward and the other downward; by $$t = 80$$ they are orbiting one another, and by $$t = 100$$ they have merged into a single, larger vortex, which persists to the end of the run.

<p align="center">
  <img src="{{ '/assets/img/posts/kh_snapshots_scalar_p10.png' | relative_url }}" alt="Passive scalar in the shear layer at six times, showing roll-up into two vortices and their pairing into one" style="width: 100%; max-width: 900px; height: auto; display: block; margin: 0 auto;">
</p>

*Figure 2: Passive scalar $$c$$ (red: fluid from the fast upper side; blue: from the lower side) in the lower shear layer at $$\mathrm{Re} = 1000$$. The fundamental rolls up into two vortices by $$t = 40$$; the subharmonic then drives them to orbit one another ($$t = 80$$) and merge ($$t = 100$$), leaving a single vortex with twice the spacing ($$t = 140$$). Only the central part of the box height is shown.*

The animation below covers the whole run.

<p align="center">
  <img src="{{ '/assets/img/posts/kh_rollup_p10.gif' | relative_url }}" alt="Animation of the passive scalar from t = 0 to t = 200: roll-up and pairing" style="width: 100%; max-width: 760px; height: auto; display: block; margin: 0 auto;">
</p>

Three diagnostics make the sequence quantitative (Figure 3). The modal amplitudes (Figure 3a) show the two stages as two exponential phases: the fundamental grows first and saturates near $$t = 49$$, and the subharmonic, which has grown in its shadow, overtakes it and saturates near $$t = 84$$, the moment of pairing. Note that in this unforced run the measured early growth rates, $$0.181$$ for the fundamental and $$0.133$$ for the subharmonic, fall below the Orr–Sommerfeld values ($$0.188$$ and $$0.147$$). We attribute the difference to the viscous spreading of the base profile, which is no longer held fixed: by $$t = 20$$ the momentum thickness has already grown by 2.7 percent, and a thicker layer grows more slowly (we have not quantified this attribution further). The forced runs of Section 4 remove exactly this effect.

The momentum thickness, $$\theta = \int (1 - \bar u^2)/4 \, dy$$ with $$\bar u$$ the streamwise average of $$u$$ ($$\theta_0 = 1/2$$ for Eq. (1)), measures the layer itself (Figure 3b). It grows in two steps, as Winant and Browand {% cite winant1974vortex %} observed in the laboratory: to between $$1.5\,\theta_0$$ and $$1.8\,\theta_0$$ at roll-up and, after a brief overshoot to $$4.6\,\theta_0$$ at $$t = 92$$ during the merger, to a mean of $$3.4\,\theta_0$$ once the paired vortex has formed (after pairing, $$\theta$$ oscillates between $$3.0\,\theta_0$$ and $$3.8\,\theta_0$$, which we attribute to the rotation of the elongated merged vortex). In other words, the layer grows not continuously but by pairing, and in this box, which allows only one pairing, it stops growing once the pairing is complete.

The third diagnostic is a consistency check. In an unforced periodic box, the kinetic energy $$E$$ can only decay through viscosity, at the rate $$dE/dt = -2\nu Z$$, where $$Z$$ is the enstrophy (half the mean-square vorticity). The relative residual of this balance stays below $$5.2 \times 10^{-7}$$ throughout the run (Figure 3c), i.e., at the level of the finite-difference estimate of $$dE/dt$$ from the saved time series; the total energy falls by 1.6 percent over the run. A second run on a grid twice as coarse ($$256 \times 512$$) reproduces the momentum thickness to within $$0.002\,\theta_0$$ and the energy to within $$4 \times 10^{-7}$$ (relative) at every instant, so the main run is resolved; the flow at $$\mathrm{Re} = 1000$$ is smooth enough that $$256 \times 512$$ points would already have sufficed.

<p align="center">
  <img src="{{ '/assets/img/posts/kh_diagnostics_p10.svg' | relative_url }}" alt="Modal amplitudes, momentum thickness and energy-budget residual against time" style="width: 100%; max-width: 980px; height: auto; display: block; margin: 0 auto;">
</p>

*Figure 3: (a) Amplitudes of the fundamental and subharmonic Fourier modes of the vorticity, with the Orr–Sommerfeld growth rates for comparison (dashed and dotted lines). (b) Momentum thickness of the layer, relative to its initial value, on the two grids. (c) Relative residual of the energy balance $$dE/dt = -2\nu Z$$.*


## 6. Limitations

The comparison is clean because the problem is idealized, and the idealizations are worth stating plainly.

- The flow is two-dimensional. Real mixing layers develop streamwise vortices and become three-dimensional and turbulent after the first pairings; nothing here addresses that transition.
- The layer is temporal, not spatial. A laboratory mixing layer grows downstream from a splitter plate, and its instability is convective; the temporal growth rates computed here correspond to the spatial ones only approximately (through Gaster's transformation, which we did not apply).
- The periodic box selects the wavelengths. With two fundamental wavelengths in the box, only one pairing is possible, and the timing of that pairing depends on the amplitude and phase we gave the subharmonic.
- A single Reynolds number ($$\mathrm{Re} = 1000$$) was studied. The viscous correction to the growth rate is small at this value, so the post does not explore the regime where viscosity changes the picture.

## Code

The complete code is in the site repository under [`code/p10_kelvin_helmholtz`](https://github.com/abelokoj/abelokoj.github.io/tree/main/code/p10_kelvin_helmholtz). It consists of `kh_core.py` (the Chebyshev eigenvalue solvers and the pseudo-spectral solver), `kh_linear.py` (Figure 1, about six minutes on a laptop), `kh_rollup.py` (the nonlinear runs, about half an hour at $$512 \times 1024$$) and `plot_kh.py`, which redraws every figure from the saved data. The folder's README lists the commands and how to send a finer run to a cluster.

## References

The tanh-profile stability problem and its inviscid growth rates are due to Michalke {% cite michalke1964inviscid %}; vortex pairing as the mechanism of mixing-layer growth was identified experimentally by Winant and Browand {% cite winant1974vortex %}, and Ho and Huerre {% cite ho1984perturbed %} review the subject. The Chebyshev methods follow Trefethen {% cite trefethen2000spectral %}, and the dealiasing rule is Orszag's {% cite orszag1971elimination %}.

{% bibliography --cited --file blog_references %}
