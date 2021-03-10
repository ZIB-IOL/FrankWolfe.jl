
include("activate.jl")
using LinearAlgebra
import Random

using MultivariatePolynomials
using DynamicPolynomials

import ReverseDiff

const N = 9

DynamicPolynomials.@polyvar X[1:9]

const max_degree = 3

const var_monomials = MultivariatePolynomials.monomials(X, 0:max_degree)

Random.seed!(42)
const all_coeffs = map(var_monomials) do m
    d = MultivariatePolynomials.degree(m)
    10 * rand() .* (rand() .> 0.3 * d / max_degree)
end

const true_poly = dot(all_coeffs, var_monomials)

function evaluate_poly(coefficients)
    poly = dot(coefficients, var_monomials)
    function p(x)
        MultivariatePolynomials.subs(poly, Pair(X, x)).a[1]
    end
end

const training_data = map(1:1000) do _
    x = 3 * randn(N)
    y = MultivariatePolynomials.subs(true_poly, Pair(X, x)) + 2 * randn()
    (x, y.a[1])
end

function f(coefficients)
    poly = evaluate_poly(coefficients)
    return 0.5 / length(training_data) * sum(training_data) do (x, y)
        (poly(x) - y)^2 
    end
end

# faster version of the objective
function f2(coefficients)
    res = 0.0
    @inbounds for (x, y) in training_data
        yhat = 0.0
        for midx in eachindex(var_monomials)
            m = var_monomials[midx]
            c = coefficients[midx]
            if c > 0
                r = c
                for j in eachindex(m.z)
                    r *= x[j]^m.z[j]
                end
                yhat += r
            end
        end
        res += (yhat - y)^2
    end
    return res * 0.5 / length(training_data)
end

function grad!(storage, p_coefficients)
    storage .= 0
    for (x, y) in training_data
        p_i = zero(eltype(p_coefficients))
        for midx in eachindex(var_monomials)
            m = var_monomials[midx]
            c = p_coefficients[midx]
            if c > 0
                r = c
                for j in eachindex(m.z)
                    r *= x[j]^m.z[j]
                end
                p_i += r
            end
        end
        for midx in eachindex(p_coefficients)
            m = var_monomials[midx]
            r = one(eltype(p_coefficients))
            for j in eachindex(m.z)
                r *= x[j]^m.z[j]
            end
            storage[midx] += r *  (p_i - y)
        end
    end
    storage ./= length(training_data)
    return nothing
end

# gradient descent
gradient = similar(all_coeffs)
xgd = rand(length(all_coeffs))
for iter in 1:2000
    if mod(iter, 50) == 0
        @info f2(xgd)
    end
    global xgd
    grad!(gradient, xgd)
    @. xgd -= 0.00001 * gradient
end

lmo = FrankWolfe.KSparseLMO(
    round(Int, length(all_coeffs) / 4),
    1.1 * maximum(all_coeffs)
)

x00 = FrankWolfe.compute_extreme_point(lmo, rand(length(all_coeffs)))

k = 1000

x0 = deepcopy(x00)

# vanilla FW
@time x, v, primal, dual_gap, trajectoryFw = FrankWolfe.fw(
    f,
    grad!,
    lmo,
    x0,
    max_iteration=k,
    line_search=FrankWolfe.adaptive,
    L=2,
    print_iter=k/50,
    emphasis=FrankWolfe.memory,
    verbose=true,
    trajectory=false,
    gradient=gradient,
);

x0 = deepcopy(x00)

@time x, v, primal, dual_gap, trajectoryFw = FrankWolfe.afw(
    f,
    grad!,
    lmo,
    x0,
    max_iteration=k,
    line_search=FrankWolfe.adaptive,
    L=2,
    print_iter=k / 50,
    emphasis=FrankWolfe.memory,
    verbose=true,
    trajectory=false,
    gradient=gradient,
);

x0 = deepcopy(x00)


@time x, v, primal, dual_gap, trajectoryBCG = FrankWolfe.bcg(
    f,
    grad!,
    lmo,
    x0,
    max_iteration=1000,
    line_search=FrankWolfe.backtracking,
    print_iter=k / 10,
    emphasis=FrankWolfe.memory,
    L=2,
    verbose=true,
    trajectory=false,
    Ktolerance=0.95,
    goodstep_tolerance=0.95,
    weight_purge_threshold=1e-10,
)
