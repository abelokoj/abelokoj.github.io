#=
Stiff ODE demo: Van der Pol oscillator with large mu, comparing an explicit
solver (Tsit5, Julia's typical RK45-equivalent default) against an
implicit/stiff-aware solver (RadauIIA5).

Julia equivalent of post2_stiff_ode_demo.py

Required packages:
    using Pkg
    Pkg.add(["OrdinaryDiffEq", "Plots", "Printf"])
=#

using OrdinaryDiffEq
using Plots
using Printf

function vdp!(du, u, p, t)
    mu = p[1]
    x, v = u[1], u[2]
    du[1] = v
    du[2] = mu * ((1 - x^2) * v - x)
end

mu = 100.0
y0 = [2.0, 0.0]
tspan = (0.0, 300.0)
t_eval = range(tspan[1], tspan[2], length=3000)

prob = ODEProblem(vdp!, y0, tspan, [mu])
t0 = time()
sol = solve(prob, RadauIIA5(), reltol=1e-6, abstol=1e-9, saveat=t_eval)
t1 = time()
@printf("Radau (implicit, stiff)     success=%s nfev=%7d nsteps=%6d wall=%.3fs\n",
        sol.retcode == :Success, sol.stats.nf, length(sol.t), t1 - t0)

# ---- Solution trajectory plot ----
p1 = plot(sol.t, [u[1] for u in sol.u], lw=1.2, legend=false,
          xlabel="time t", ylabel="x(t)",
          title="Van der Pol oscillator, mu=$(mu) (relaxation oscillation)")
savefig(p1, "vdp_trajectory.png")

# ---- Phase portrait ----
p2 = plot([u[1] for u in sol.u], [u[2] for u in sol.u], lw=0.8, legend=false,
          xlabel="x", ylabel="v = dx/dt", title="Phase portrait (limit cycle)",
          size=(550, 520))
savefig(p2, "vdp_phase.png")

# ---- Cost comparison across mu (short fixed horizon, no artificial step cap) ----
mus = [1, 10, 50, 100, 300, 600, 1000, 2000]
cost_explicit_nfev = Int[]
cost_implicit_nfev = Int[]
cost_explicit_steps = Int[]
cost_implicit_steps = Int[]

for m in mus
    tspan_m = (0.0, 4.0)

    prob_e = ODEProblem(vdp!, y0, tspan_m, [Float64(m)])
    sol_e = solve(prob_e, Tsit5(), reltol=1e-6, abstol=1e-9)
    push!(cost_explicit_nfev, sol_e.stats.nf)
    push!(cost_explicit_steps, length(sol_e.t))

    prob_i = ODEProblem(vdp!, y0, tspan_m, [Float64(m)])
    sol_i = solve(prob_i, RadauIIA5(), reltol=1e-6, abstol=1e-9)
    push!(cost_implicit_nfev, sol_i.stats.nf)
    push!(cost_implicit_steps, length(sol_i.t))

    @printf("mu=%5d  Tsit5 nfev=%8d steps=%6d  Radau nfev=%6d steps=%5d\n",
            m, cost_explicit_nfev[end], cost_explicit_steps[end],
            cost_implicit_nfev[end], cost_implicit_steps[end])
end

p3 = plot(mus, cost_explicit_nfev, marker=:circle, yscale=:log10,
          label="Tsit5 (explicit)", xlabel="stiffness parameter mu",
          ylabel="number of RHS evaluations (log scale)",
          title="Solver cost vs. stiffness, fixed accuracy tolerance")
plot!(p3, mus, cost_implicit_nfev, marker=:square, label="RadauIIA5 (implicit)")
savefig(p3, "cost_vs_stiffness.png")

open("results.txt", "w") do fh
    for (m, ne, ni) in zip(mus, cost_explicit_nfev, cost_implicit_nfev)
        @printf(fh, "mu=%d nfev_explicit=%d nfev_implicit=%d\n", m, ne, ni)
    end
end

println("done")
