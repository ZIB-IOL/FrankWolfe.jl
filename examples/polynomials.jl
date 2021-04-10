
include("activate.jl")
using LinearAlgebra
import Random

using MultivariatePolynomials
using DynamicPolynomials

import ReverseDiff
using FiniteDifferences
import JSON

const N = 15

DynamicPolynomials.@polyvar X[1:15]

const max_degree = 4

const var_monomials = MultivariatePolynomials.monomials(X, 0:max_degree)

Random.seed!(42)
const all_coeffs = map(var_monomials) do m
    d = MultivariatePolynomials.degree(m)
    return 10 * rand() .* (rand() .> 0.7 * d / max_degree)
end

const true_poly = dot(all_coeffs, var_monomials)

function evaluate_poly(coefficients)
    poly = dot(coefficients, var_monomials)
    return function p(x)
        return MultivariatePolynomials.subs(poly, Pair(X, x)).a[1]
    end
end

const training_data = map(1:500) do _
    x = 0.1 * randn(N)
    y = MultivariatePolynomials.subs(true_poly, Pair(X, x)) + 2 * randn()
    return (x, y.a[1])
end

const extended_training_data = map(training_data) do (x, y)
    x_ext = getproperty.(MultivariatePolynomials.subs.(var_monomials, X => x), :α)
    return (x_ext, y)
end

const test_data = map(1:1000) do _
    x = 0.4 * randn(N)
    y = MultivariatePolynomials.subs(true_poly, Pair(X, x)) + 2 * randn()
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
        (all_coeffs[idx] - coeffs[idx])^2
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
    function callback(state)
        push!(
            trajectory_arr,
            (Tuple(state)[1:5]..., f_test(state.x), coefficient_errors(state.x))
        )
    end
end

#Check the gradient using finite differences just in case
gradient = similar(all_coeffs)
FrankWolfe.check_gradients(grad!, f, gradient)

max_iter = 50_000

xgd = rand(length(all_coeffs))


lmo = FrankWolfe.LpNormLMO{1}(norm(all_coeffs))

# L estimate
num_pairs = 10000
L_estimate = -Inf
gradient_aux = similar(gradient)
for i in 1:num_pairs
    global L_estimate
    x = compute_extreme_point(lmo, randn(size(xgd)))
    y = compute_extreme_point(lmo, randn(size(xgd)))
    grad!(gradient, x)
    grad!(gradient_aux, y)
    new_L = norm(gradient - gradient_aux) / norm(x - y)
    if new_L > L_estimate
        L_estimate = new_L
    end
end

# gradient descent

xgd = rand(length(all_coeffs))
training_gd = Float64[]
test_gd = Float64[]
coeff_error = Float64[]
time_start = time_ns()
gd_times = Float64[]
for iter in 1:max_iter
    global xgd
    grad!(gradient, xgd)
    @. xgd -= gradient / L_estimate
    push!(training_gd, f(xgd))
    push!(test_gd, f_test(xgd))
    push!(coeff_error, coefficient_errors(xgd))
    push!(gd_times, (time_ns() - time_start) * 1e-9)
end

@info "Gradient descent training loss $(f(xgd))"
@info "Gradient descent test loss $(f_test(xgd))"
@info "Coefficient error $(coefficient_errors(xgd))"


x00 = FrankWolfe.compute_extreme_point(lmo, rand(length(all_coeffs)))

x0 = deepcopy(x00)

# vanilla FW
trajectory_fw = []
callback = build_callback(trajectory_fw)
@time x_fw, v, primal, dual_gap, _ = FrankWolfe.frank_wolfe(
    f,
    grad!,
    lmo,
    x0,
    max_iteration=max_iter,
    line_search=FrankWolfe.adaptive,
    print_iter=max_iter ÷ 10,
    emphasis=FrankWolfe.memory,
    verbose=true,
    gradient=gradient,
    callback=callback,
);

@info "Vanilla training loss $(f(x_fw))"
@info "Test loss $(f_test(x_fw))"
@info "Coefficient error $(coefficient_errors(x_fw))"

trajectory_bcg = []
callback = build_callback(trajectory_bcg)

x0 = deepcopy(x00)
@time x_bcg, v, primal, dual_gap, _ = FrankWolfe.blended_conditional_gradient(
    f,
    grad!,
    lmo,
    x0,
    max_iteration=max_iter,
    line_search=FrankWolfe.adaptive,
    print_iter=max_iter ÷ 10,
    emphasis=FrankWolfe.memory,
    verbose=true,
    weight_purge_threshold=1e-10,
    callback=callback,
)

@info "BCG training loss $(f(x_bcg))"
@info "Test loss $(f_test(x_bcg))"
@info "Coefficient error $(coefficient_errors(x_bcg))"

open(joinpath(@__DIR__, "polynomial_result.json"), "w") do f
    data = JSON.json(
        (
            trajectory_arr_fw=trajectory_fw,
            trajectory_arr_bcg=trajectory_bcg,
            function_values_gd=training_gd,
            function_values_test_gd=test_gd,
            coefficient_error_gd=coeff_error,
            gd_times=gd_times,
        )
    )
    write(f, data)
end
