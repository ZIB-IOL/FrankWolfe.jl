
include("activate.jl")
using LinearAlgebra
import Random

using MultivariatePolynomials
using DynamicPolynomials

import ReverseDiff
using FiniteDifferences

const N = 9

DynamicPolynomials.@polyvar X[1:9]

const max_degree = 3

const var_monomials = MultivariatePolynomials.monomials(X, 0:max_degree)

Random.seed!(42)
const all_coeffs = map(var_monomials) do m
    d = MultivariatePolynomials.degree(m)
    return 10 * rand() .* (rand() .> 0.6 * d / max_degree)
end

const true_poly = dot(all_coeffs, var_monomials)

function evaluate_poly(coefficients)
    poly = dot(coefficients, var_monomials)
    return function p(x)
        return MultivariatePolynomials.subs(poly, Pair(X, x)).a[1]
    end
end

const training_data = map(1:1000) do _
    x = 3 * randn(N)
    y = MultivariatePolynomials.subs(true_poly, Pair(X, x)) + 2 * randn()
    return (x, y.a[1])
end

const extended_training_data = map(training_data) do (x, y)
    x_ext = getproperty.(MultivariatePolynomials.subs.(var_monomials, X => x), :α)
    return (x_ext, y)
end

const test_data = map(1:1000) do _
    x = 3 * randn(N)
    y = MultivariatePolynomials.subs(true_poly, Pair(X, x)) + 2 * randn()
    return (x, y.a[1])
end

const extended_test_data = map(test_data) do (x, y)
    x_ext = getproperty.(MultivariatePolynomials.subs.(var_monomials, X => x), :α)
    return (x_ext, y)
end

function f(coefficients)
    poly = evaluate_poly(coefficients)
    return 0.5 / length(training_data) * sum(training_data) do (x, y)
        return (poly(x) - y)^2
    end
end

# extended version, much faster (memory intense)
function f3(coefficients)
    return 0.5 / length(extended_training_data) * sum(extended_training_data) do (x, y)
        return (dot(coefficients, x) - y)^2
    end
end

function f_test(coefficients)
    return 0.5 / length(extended_test_data) * sum(extended_test_data) do (x, y)
        return (dot(coefficients, x) - y)^2
    end
end

function matching_zeros(coeffs)
    return count(eachindex(all_coeffs)) do idx
        return all_coeffs[idx] ≈ 0 && coeffs[idx] ≈ 0
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

#Check the gradient using finite differences just in case
gradient = similar(all_coeffs)
FrankWolfe.check_gradients(grad!, f3, gradient)

# gradient descent
xgd = rand(length(all_coeffs))
for iter in 1:3000
    global xgd
    grad!(gradient, xgd)
    @. xgd -= 0.00001 * gradient
end

@info "Gradient descent training loss $(f3(xgd))"
@info "Gradient descent test loss $(f_test(xgd))"
@info "Matching zeros $(matching_zeros(xgd))"

lmo = FrankWolfe.KSparseLMO(round(Int, length(all_coeffs) / 4), 1.1 * maximum(all_coeffs))

x00 = FrankWolfe.compute_extreme_point(lmo, rand(length(all_coeffs)))

k = 1e5

x0 = deepcopy(x00)

# vanilla FW
@time x, v, primal, dual_gap, trajectoryFw = FrankWolfe.frank_wolfe(
    f3,
    grad!,
    lmo,
    x0,
    max_iteration=k,
    line_search=FrankWolfe.adaptive,
    print_iter=k / 10,
    emphasis=FrankWolfe.memory,
    verbose=true,
    trajectory=false,
    gradient=gradient,
);

@info "Vanilla training loss $(f3(x))"
@info "Test loss $(f_test(x))"
@info "Matching zeros $(matching_zeros(x))"

x0 = deepcopy(x00)

@time x, v, primal, dual_gap, trajectoryFw = FrankWolfe.away_frank_wolfe(
    f3,
    grad!,
    lmo,
    x0,
    max_iteration=k,
    line_search=FrankWolfe.adaptive,
    print_iter=k / 10,
    emphasis=FrankWolfe.memory,
    verbose=true,
    lazy=true,
    trajectory=false,
    gradient=gradient,
);

@info "AFW training loss $(f3(x))"
@info "Test loss $(f_test(x))"
@info "Matching zeros $(matching_zeros(x))"

x0 = deepcopy(x00)
@time x, v, primal, dual_gap, trajectoryBCG = FrankWolfe.blended_conditional_gradient(
    f3,
    grad!,
    lmo,
    x0,
    max_iteration=k,
    line_search=FrankWolfe.adaptive,
    print_iter=k / 10,
    emphasis=FrankWolfe.memory,
    verbose=true,
    trajectory=false,
    weight_purge_threshold=1e-10,
)

@info "BCG training loss $(f3(x))"
@info "Test loss $(f_test(x))"
@info "Matching zeros $(matching_zeros(x))"
