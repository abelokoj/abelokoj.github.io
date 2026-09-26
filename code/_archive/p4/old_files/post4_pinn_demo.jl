#=
Physics-informed neural network (PINN) vs. a classical solver, on a forced
damped first-order linear ODE with a closed-form exact solution.

Julia equivalent of post4_pinn_demo.py — a small single-hidden-layer tanh
network trained by minimizing an ODE-residual + initial-condition loss,
using Optim.jl's L-BFGS (finite-difference gradient, matching scipy's
default behavior when no analytic gradient is supplied), compared against
OrdinaryDiffEq's Tsit5 and the exact analytic solution.

Required packages:
    using Pkg
    Pkg.add(["OrdinaryDiffEq", "Optim", "Plots", "Printf", "Random"])
=#

using OrdinaryDiffEq
using Optim
using Random
using Plots
using Printf

# ---- Problem definition ----
const k = 2.0
const A = 1.0
const omega = 3.0
const y0_val = 0.3
const T = 3.0

rhs(y, t) = -k * (y - A * sin(omega * t))

function exact_solution(t)
    homog_coeff = y0_val - (k * A / (k^2 + omega^2)) * (-omega)
    part = (k * A / (k^2 + omega^2)) * (k * sin(omega * t) - omega * cos(omega * t))
    return homog_coeff * exp(-k * t) + part
end

@assert abs(exact_solution(0.0) - y0_val) < 1e-10

# ---- Classical solver baseline ----
t_dense = collect(range(0, T, length=400))

function ode!(du, u, p, t)
    du[1] = rhs(u[1], t)
end

prob = ODEProblem(ode!, [y0_val], (0.0, T))
t0 = time()
sol_classical = solve(prob, Tsit5(), reltol=1e-10, abstol=1e-12, saveat=t_dense)
t1 = time()
classical_time = t1 - t0
y_classical = [u[1] for u in sol_classical.u]
y_exact_dense = exact_solution.(t_dense)
classical_err = maximum(abs.(y_classical .- y_exact_dense))
@printf("Classical Tsit5: wall=%.3f ms, max err vs exact = %.2e\n",
        classical_time * 1000, classical_err)

# ---- PINN setup ----
const n_hidden = 12
const n_params = 3 * n_hidden + 1   # w1, b1, w2 (n_hidden each) + b2

function unpack(theta)
    w1 = theta[1:n_hidden]
    b1 = theta[n_hidden+1:2*n_hidden]
    w2 = theta[2*n_hidden+1:3*n_hidden]
    b2 = theta[3*n_hidden+1]
    return w1, b1, w2, b2
end

function y_hat(t::AbstractVector, theta)
    w1, b1, w2, b2 = unpack(theta)
    z = t * w1' .+ b1'              # (N, H)
    h = tanh.(z)
    return h * w2 .+ b2             # (N,)
end

function dy_hat_dt(t::AbstractVector, theta)
    w1, b1, w2, b2 = unpack(theta)
    z = t * w1' .+ b1'
    dh = (1 .- tanh.(z) .^ 2) .* w1'   # derivative of tanh(w1 t + b1) wrt t
    return dh * w2
end

function loss(theta, t_colloc)
    y = y_hat(t_colloc, theta)
    dy = dy_hat_dt(t_colloc, theta)
    residual = dy .- rhs.(y, t_colloc)
    ic_pred = y_hat([0.0], theta)[1]
    return mean(residual .^ 2) + 50.0 * (ic_pred - y0_val)^2
end
mean(v) = sum(v) / length(v)

rng = MersenneTwister(42)
t_colloc = collect(range(0, T, length=60))
theta0 = 0.5 .* randn(rng, n_params)

loss_closure = theta -> loss(theta, t_colloc)

t0 = time()
result = optimize(loss_closure, theta0, LBFGS(),
                   Optim.Options(iterations=4000, f_tol=1e-14, g_tol=1e-12))
t1 = time()
pinn_time = t1 - t0
theta_star = Optim.minimizer(result)
@printf("PINN training: wall=%.1f ms, final loss=%.3e, iters=%d, converged=%s\n",
        pinn_time * 1000, Optim.minimum(result), Optim.iterations(result),
        Optim.converged(result))

y_pinn_dense = y_hat(t_dense, theta_star)
pinn_err = maximum(abs.(y_pinn_dense .- y_exact_dense))
@printf("PINN: max err vs exact = %.2e\n", pinn_err)

# ---- Plot 1: solution comparison ----
p1 = plot(t_dense, y_exact_dense, lw=2.5, color=:black, label="exact analytic solution",
          xlabel="time t", ylabel="y(t)",
          title="Forced damped ODE: exact vs. classical solver vs. PINN")
plot!(p1, t_dense, y_classical, lw=1.8, linestyle=:dash,
      label=@sprintf("classical Tsit5 (err=%.1e)", classical_err))
plot!(p1, t_dense, y_pinn_dense, lw=2.2, linestyle=:dot,
      label=@sprintf("PINN (err=%.1e)", pinn_err))
savefig(p1, "solution_comparison.png")

# ---- Plot 2: cost/accuracy tradeoff summary ----
p2 = scatter([classical_time * 1000], [classical_err], markersize=9,
             label="Classical Tsit5", xscale=:log10, yscale=:log10,
             xlabel="wall-clock time (ms, log scale)",
             ylabel="max error vs. exact solution (log scale)",
             title="Accuracy vs. compute cost")
scatter!(p2, [pinn_time * 1000], [pinn_err], markersize=9, marker=:utriangle,
         label="PINN (L-BFGS training)")
savefig(p2, "cost_accuracy.png")

open("results.txt", "w") do fh
    @printf(fh, "classical_time_ms=%.4f classical_err=%.3e\n",
            classical_time * 1000, classical_err)
    @printf(fh, "pinn_time_ms=%.4f pinn_err=%.3e pinn_iters=%d\n",
            pinn_time * 1000, pinn_err, Optim.iterations(result))
end

println("done")
