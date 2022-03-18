using FrankWolfe
using ProgressMeter
using Arpack
using Plots
using DoubleFloats
using ReverseDiff

using LinearAlgebra
import Random

using MultivariatePolynomials
using DynamicPolynomials

using FiniteDifferences
import JSON
using Statistics

const N = 15

DynamicPolynomials.@polyvar X[1:15]

const max_degree = 4
coefficient_magnitude = 10
noise_magnitude = 1

const var_monomials = MultivariatePolynomials.monomials(X, 0:max_degree)

Random.seed!(42)
all_coeffs = map(var_monomials) do m
    d = MultivariatePolynomials.degree(m)
    return coefficient_magnitude * rand()
end
random_vector = rand(length(all_coeffs))
cutoff = quantile(random_vector, 0.95)
all_coeffs[findall(<(cutoff), random_vector)]  .= 0.0

const true_poly = dot(all_coeffs, var_monomials)

function evaluate_poly(coefficients)
    poly = dot(coefficients, var_monomials)
    return function p(x)
        return MultivariatePolynomials.subs(poly, Pair(X, x)).a[1]
    end
end

const training_data = map(1:500) do _
    x = 0.1 * randn(N)
    y = MultivariatePolynomials.subs(true_poly, Pair(X, x)) + noise_magnitude * randn()
    return (x, y.a[1])
end

const extended_training_data = map(training_data) do (x, y)
    x_ext = getproperty.(MultivariatePolynomials.subs.(var_monomials, X => x), :α)
    return (x_ext, y)
end

const test_data = map(1:1000) do _
    x = 0.4 * randn(N)
    y = MultivariatePolynomials.subs(true_poly, Pair(X, x)) + noise_magnitude * randn()
    return (x, y.a[1])
end

const extended_test_data = map(test_data) do (x, y)
    x_ext = getproperty.(MultivariatePolynomials.subs.(var_monomials, X => x), :α)
    return (x_ext, y)
end

function f(coefficients)
    return 0.5 / length(extended_training_data) * sum(extended_training_data) do (x, y)
        return (dot(coefficients, x) - y)^2
    end
end

function f_test(coefficients)
    return 0.5 / length(extended_test_data) * sum(extended_test_data) do (x, y)
        return (dot(coefficients, x) - y)^2
    end
end

function coefficient_errors(coeffs)
    return 0.5 * sum(eachindex(all_coeffs)) do idx
        return (all_coeffs[idx] - coeffs[idx])^2
    end
end

function grad!(storage, coefficients)
    storage .= 0
    for (x, y) in extended_training_data
        p_i = dot(coefficients, x) - y
        @. storage += x * p_i
    end
    storage ./= length(training_data)
    return nothing
end

function build_callback(trajectory_arr)
    return function callback(state)
        return push!(
            trajectory_arr,
            (Tuple(state)[1:5]..., f_test(state.x), coefficient_errors(state.x)),
        )
    end
end

#Check the gradient using finite differences just in case
gradient = similar(all_coeffs)

max_iter = 100_000
random_initialization_vector = rand(length(all_coeffs))

#lmo = FrankWolfe.LpNormLMO{1}(100 * maximum(all_coeffs))

lmo = FrankWolfe.LpNormLMO{1}(0.95 * norm(all_coeffs, 1))

# L estimate
num_pairs = 10000
L_estimate = -Inf
gradient_aux = similar(gradient)
for i in 1:num_pairs
    global L_estimate
    x = compute_extreme_point(lmo, randn(size(all_coeffs)))
    y = compute_extreme_point(lmo, randn(size(all_coeffs)))
    grad!(gradient, x)
    grad!(gradient_aux, y)
    new_L = norm(gradient - gradient_aux) / norm(x - y)
    if new_L > L_estimate
        L_estimate = new_L
    end
end

# L1 projection
# inspired by https://github.com/MPF-Optimization-Laboratory/ProjSplx.jl
function projnorm1(x, τ)
    n = length(x)
    if norm(x, 1) ≤ τ
        return x
    end
    u = abs.(x)
    # simplex projection
    bget = false
    s_indices = sortperm(u, rev=true)
    tsum = zero(τ)

    @inbounds for i in 1:n-1
        tsum += u[s_indices[i]]
        tmax = (tsum - τ) / i
        if tmax ≥ u[s_indices[i+1]]
            bget = true
            break
        end
    end
    if !bget
        tmax = (tsum + u[s_indices[n]] - τ) / n
    end

    @inbounds for i in 1:n
        u[i] = max(u[i] - tmax, 0)
        u[i] *= sign(x[i])
    end
    return u
end

# gradient descent

xgd = FrankWolfe.compute_extreme_point(lmo, random_initialization_vector)
training_gd = Float64[]
test_gd = Float64[]
coeff_error = Float64[]
time_start = time_ns()
gd_times = Float64[]
for iter in 1:max_iter
    global xgd
    grad!(gradient, xgd)
    xgd = projnorm1(xgd - gradient / L_estimate, lmo.right_hand_side)
    push!(training_gd, f(xgd))
    push!(test_gd, f_test(xgd))
    push!(coeff_error, coefficient_errors(xgd))
    push!(gd_times, (time_ns() - time_start) * 1e-9)
end

@info "Gradient descent training loss $(f(xgd))"
@info "Gradient descent test loss $(f_test(xgd))"
@info "Coefficient error $(coefficient_errors(xgd))"


x00 = FrankWolfe.compute_extreme_point(lmo, random_initialization_vector)
x0 = deepcopy(x00)

# lazy AFW
trajectory_lafw = []
callback = build_callback(trajectory_lafw)
@time x_lafw, v, primal, dual_gap, _ = FrankWolfe.away_frank_wolfe(
    f,
    grad!,
    lmo,
    x0,
    max_iteration=max_iter,
    line_search=FrankWolfe.Adaptive(L_est=L_estimate),
    print_iter=max_iter ÷ 10,
    memory_mode=FrankWolfe.InplaceEmphasis(),
    verbose=true,
    lazy=true,
    gradient=gradient,
    callback=callback,
);

@info "Lazy AFW training loss $(f(x_lafw))"
@info "Test loss $(f_test(x_lafw))"
@info "Coefficient error $(coefficient_errors(x_lafw))"

trajectory_bcg = []
callback = build_callback(trajectory_bcg)

x0 = deepcopy(x00)
@time x_bcg, v, primal, dual_gap, _ = FrankWolfe.blended_conditional_gradient(
    f,
    grad!,
    lmo,
    x0,
    max_iteration=max_iter,
    line_search=FrankWolfe.Adaptive(L_est=L_estimate),
    print_iter=max_iter ÷ 10,
    memory_mode=FrankWolfe.InplaceEmphasis(),
    verbose=true,
    weight_purge_threshold=1e-10,
    callback=callback,
)

@info "BCG training loss $(f(x_bcg))"
@info "Test loss $(f_test(x_bcg))"
@info "Coefficient error $(coefficient_errors(x_bcg))"


x0 = deepcopy(x00)

#  compute reference solution using lazy AFW
trajectory_lafw_ref = []
callback = build_callback(trajectory_lafw_ref)
@time _, _, primal_ref, _, _ = FrankWolfe.away_frank_wolfe(
    f,
    grad!,
    lmo,
    x0,
    max_iteration=2 * max_iter,
    line_search=FrankWolfe.Adaptive(L_est=L_estimate),
    print_iter=max_iter ÷ 10,
    memory_mode=FrankWolfe.InplaceEmphasis(),
    verbose=true,
    lazy=true,
    gradient=gradient,
    callback=callback,
);

open(joinpath(@__DIR__, "polynomial_result.json"), "w") do f
    data = JSON.json((
        trajectory_arr_lafw=trajectory_lafw,
        trajectory_arr_bcg=trajectory_bcg,
        function_values_gd=training_gd,
        function_values_test_gd=test_gd,
        coefficient_error_gd=coeff_error,
        gd_times=gd_times,
        ref_primal_value=primal_ref,
    ))
    return write(f, data)
end

#Count missing\extra terms
print("\n Number of extra terms in GD: ", sum((all_coeffs .== 0) .* (xgd .!= 0)))
print("\n Number of missing terms in GD: ", sum((all_coeffs .!= 0) .* (xgd .== 0)))

print("\n Number of extra terms in BCG: ", sum((all_coeffs .== 0) .* (x_bcg .!= 0)))
print("\n Number of missing terms in BCG: ", sum((all_coeffs .!= 0) .* (x_bcg .== 0)))

print("\n Number of missing terms in Lazy AFW: ", sum((all_coeffs .== 0) .* (x_lafw .!= 0)))
print("\n Number of extra terms in Lazy AFW: ", sum((all_coeffs .!= 0) .* (x_lafw .== 0)))
