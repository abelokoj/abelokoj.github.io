#=
Sherman-Morrison accelerated implicit Euler for a mass-varying linear ODE system,
motivated by sustainable-aviation fuel-burn dynamics.

Julia equivalent of sherman_morrison_demo_p1.py -- same model, same
benchmark structure, same correctness check.

Required packages:
    using Pkg
    Pkg.add(["Plots", "Printf"])
=#

using LinearAlgebra
using Random
using Printf
using Plots

# --- Publication-quality plotting defaults ---
default(fontfamily="Computer Modern", guidefontsize=13, titlefontsize=13,
        legendfontsize=10, tickfontsize=11, linewidth=1.4, framestyle=:box, dpi=300)

function build_system(n::Int, rng)
    A0 = -1.5 .* Matrix{Float64}(I, n, n) .+ 0.05 .* randn(rng, n, n)
    max_real = maximum(real.(eigvals(A0)))
    A0 = A0 .- (max_real + 1.0) .* Matrix{Float64}(I, n, n)
    u = randn(rng, n)
    v = randn(rng, n)
    b = randn(rng, n)
    y0 = randn(rng, n)
    return A0, u, v, b, y0
end

fuel_burn_coeff(t, T; c0=2.0) = c0 * (1.0 - t / T)

function solve_naive(A0, u, v, b, y0, h, steps, T)
    n = length(y0)
    y = copy(y0)
    traj = zeros(steps + 1, n)
    traj[1, :] .= y
    for k in 0:steps-1
        t = k * h
        c = fuel_burn_coeff(t, T)
        Ak = A0 .+ c .* (u * v')
        M = Matrix{Float64}(I, n, n) .- h .* Ak
        rhs = y .+ h .* b
        y = M \ rhs                     # full O(n^3) solve every step
        traj[k+2, :] .= y
    end
    return traj
end

function solve_sherman_morrison(A0, u, v, b, y0, h, steps, T)
    n = length(y0)
    B0 = Matrix{Float64}(I, n, n) .- h .* A0
    B0_inv = inv(B0)                    # ONE O(n^3) factorization, reused every step
    y = copy(y0)
    traj = zeros(steps + 1, n)
    traj[1, :] .= y
    for k in 0:steps-1
        t = k * h
        c = fuel_burn_coeff(t, T)
        alpha = -h * c
        Binv_u = B0_inv * u
        v_Binv = B0_inv' * v            # column vector equal to (v' * B0_inv)'
        denom = 1.0 + alpha * (v' * Binv_u)
        M_inv = B0_inv .- (alpha / denom) .* (Binv_u * v_Binv')
        rhs = y .+ h .* b
        y = M_inv * rhs                 # O(n^2) per step
        traj[k+2, :] .= y
    end
    return traj
end

# ---- 1) Correctness check ----
rng = MersenneTwister(0)
n_check = 30
A0, u, v, b, y0 = build_system(n_check, rng)
T, steps = 5.0, 400
h = T / steps
traj_naive = solve_naive(A0, u, v, b, y0, h, steps, T)
traj_sm = solve_sherman_morrison(A0, u, v, b, y0, h, steps, T)
max_err = maximum(abs.(traj_naive .- traj_sm))
@printf("Max abs difference between naive and Sherman-Morrison trajectories: %.3e\n", max_err)

# ---- 2) Trajectory plot (first 3 state components) ----
t_grid = range(0, T, length=steps + 1)
p1 = plot(title="Implicit-Euler trajectories (Sherman-Morrison solve)",
          xlabel="time t (flight horizon, normalized)", ylabel="state value")
for i in 1:3
    plot!(p1, t_grid, traj_sm[:, i], label="state y\$(i)(t)")
end
savefig(p1, "trajectory_p1.png")

# ---- 3) Timing benchmark vs. system size n ----
sizes = [20, 40, 80, 160, 320]
steps_bench = 150
T_bench = 3.0
h_bench = T_bench / steps_bench

naive_times = Float64[]
sm_times = Float64[]
for n in sizes
    A0n, un, vn, bn, y0n = build_system(n, rng)

    t0 = time()
    solve_naive(A0n, un, vn, bn, y0n, h_bench, steps_bench, T_bench)
    t1 = time()
    push!(naive_times, t1 - t0)

    t0 = time()
    solve_sherman_morrison(A0n, un, vn, bn, y0n, h_bench, steps_bench, T_bench)
    t1 = time()
    push!(sm_times, t1 - t0)

    @printf("n=%4d  naive=%.4fs  sherman-morrison=%.4fs  speedup=%.2fx\n",
            n, naive_times[end], sm_times[end], naive_times[end] / sm_times[end])
end

p2 = plot(sizes, naive_times, marker=:circle, label="Naive (refactor every step)",
          xlabel="system dimension n", ylabel="wall-clock time for $(steps_bench) steps (s)",
          title="Runtime: naive re-solve vs. Sherman-Morrison update")
plot!(p2, sizes, sm_times, marker=:square, label="Sherman-Morrison (factor once)")
savefig(p2, "timing_p1.png")

open("results.txt", "w") do fh
    @printf(fh, "max_err=%.3e\n", max_err)
    for (n, tn, ts) in zip(sizes, naive_times, sm_times)
        @printf(fh, "n=%d naive=%.5f sm=%.5f speedup=%.2f\n", n, tn, ts, tn / ts)
    end
end

println("done")
