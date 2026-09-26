#=
Polynomial Chaos Expansion (PCE) surrogate modeling demo.

Julia equivalent of post3_pce_demo.py — Legendre-basis PCE built via
Gauss-Legendre quadrature projection, compared against Monte Carlo for
convergence of mean/variance estimates.

Required packages:
    using Pkg
    Pkg.add(["FastGaussQuadrature", "Plots", "Printf", "Statistics"])
=#

using FastGaussQuadrature
using Plots
using Printf
using Statistics
using Random

f(x) = 1.0 / (1.0 + 25.0 * x^2)

# ---- Legendre basis via the standard three-term recurrence ----
# P_0(x) = 1, P_1(x) = x, n P_n(x) = (2n-1) x P_{n-1}(x) - (n-1) P_{n-2}(x)
function legendre_basis_eval(order::Int, x::AbstractVector)
    n = length(x)
    Phi = zeros(n, order + 1)
    Phi[:, 1] .= 1.0
    if order >= 1
        Phi[:, 2] .= x
    end
    for j in 2:order
        Phi[:, j+1] .= ((2j - 1) .* x .* Phi[:, j] .- (j - 1) .* Phi[:, j-1]) ./ j
    end
    return Phi
end

function pce_coefficients(order::Int, f)
    n_quad = order + 5
    nodes, weights = gausslegendre(n_quad)
    fvals = f.(nodes)
    Phi = legendre_basis_eval(order, nodes)
    coeffs = zeros(order + 1)
    for j in 0:order
        norm_j = 2.0 / (2j + 1)
        coeffs[j+1] = sum(weights .* fvals .* Phi[:, j+1]) / norm_j
    end
    return coeffs
end

pce_eval(coeffs::AbstractVector, x::AbstractVector) =
    legendre_basis_eval(length(coeffs) - 1, x) * coeffs

# ---- 1) Surrogate accuracy vs. order ----
x_fine = collect(range(-1, 1, length=500))
f_true = f.(x_fine)

orders_to_plot = [2, 4, 8, 14]
p1 = plot(x_fine, f_true, lw=2.2, color=:black, label="true f(x)",
          xlabel="x", ylabel="f(x)",
          title="Legendre polynomial chaos surrogate vs. true response")
for order in orders_to_plot
    coeffs = pce_coefficients(order, f)
    f_approx = pce_eval(coeffs, x_fine)
    plot!(p1, x_fine, f_approx, lw=1.3, label="PCE order $(order)")
end
savefig(p1, "surrogate_fit.png")

# ---- 2) Mean/variance convergence: PCE vs Monte Carlo ----
coeffs_ref = pce_coefficients(40, f)
mean_ref = coeffs_ref[1]                        # E[f] = c_0
var_ref = sum((coeffs_ref[2:end] .^ 2) .* (2.0 ./ (2 .* (1:length(coeffs_ref)-1) .+ 1)) ./ 2.0)
@printf("Reference mean = %.6f, reference variance = %.6f\n", mean_ref, var_ref)

pce_orders = collect(1:15)
pce_mean_err = Float64[]
pce_var_err = Float64[]
pce_neval = Int[]
for order in pce_orders
    c = pce_coefficients(order, f)
    mean_est = c[1]
    var_est = sum((c[2:end] .^ 2) .* (2.0 ./ (2 .* (1:length(c)-1) .+ 1)) ./ 2.0)
    push!(pce_mean_err, abs(mean_est - mean_ref))
    push!(pce_var_err, abs(var_est - var_ref))
    push!(pce_neval, order + 5)
end

mc_sizes = [10, 30, 100, 300, 1000, 3000, 10000, 30000]
mc_mean_err = Float64[]
mc_var_err = Float64[]
n_repeats = 30
rng = MersenneTwister(1)
for N in mc_sizes
    mean_errs = Float64[]
    var_errs = Float64[]
    for _ in 1:n_repeats
        x_samp = rand(rng, N) .* 2.0 .- 1.0     # Uniform(-1, 1)
        fvals = f.(x_samp)
        push!(mean_errs, abs(mean(fvals) - mean_ref))
        push!(var_errs, abs(var(fvals, corrected=false) - var_ref))
    end
    push!(mc_mean_err, mean(mean_errs))
    push!(mc_var_err, mean(var_errs))
end

p2a = plot(pce_neval, pce_mean_err, marker=:square, xscale=:log10, yscale=:log10,
           label="PCE (quadrature evals)", xlabel="number of model evaluations",
           ylabel="|mean error|", title="Convergence of E[f] estimate")
plot!(p2a, mc_sizes, mc_mean_err, marker=:circle, label="Monte Carlo (samples)")

p2b = plot(pce_neval, pce_var_err, marker=:square, xscale=:log10, yscale=:log10,
           label="PCE (quadrature evals)", xlabel="number of model evaluations",
           ylabel="|variance error|", title="Convergence of Var[f] estimate")
plot!(p2b, mc_sizes, mc_var_err, marker=:circle, label="Monte Carlo (samples)")

p2 = plot(p2a, p2b, layout=(1, 2), size=(1100, 450))
savefig(p2, "convergence.png")

open("results.txt", "w") do fh
    @printf(fh, "mean_ref=%.6f var_ref=%.6f\n", mean_ref, var_ref)
    for (o, ne, me, ve) in zip(pce_orders, pce_neval, pce_mean_err, pce_var_err)
        @printf(fh, "order=%d nevals=%d mean_err=%.3e var_err=%.3e\n", o, ne, me, ve)
    end
end

println("done")
