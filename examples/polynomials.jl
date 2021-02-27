
include("activate.jl")
using LinearAlgebra
import Random

using MultivariatePolynomials
using TypedPolynomials

import ReverseDiff

N = 10

@polyvar X[1:N]


const max_degree = 3
Random.seed!(33)
const all_coeffs = map(1:max_degree) do d
    10 * rand(N^d) .* (rand(N^d) .> 0.8 * d / 3)
end
const flattened_coeff = pushfirst!(collect(Iterators.flatten(all_coeffs)), 12.0)


function true_poly(x)
    current = 1
    12 + sum(1:max_degree) do d
        current = [xi * c for xi in x for c in current]
        # replace whith dot the issue is fixed
        # https://github.com/JuliaAlgebra/TypedPolynomials.jl/issues/65
        sum(c * x for (c, x) in zip(all_coeffs[d], current))
    end
end

t = true_poly(X)

# coefficients are considered flattened
# builds the polynomial function
function evaluate_poly(coefficients)
    function p(x)
        current = 1
        res = coefficients[1]
        current_idx = 2
        @inbounds for d in 1:max_degree
            current = [xi * c for xi in x for c in current]
            for idx in eachindex(current)
                res += coefficients[current_idx+idx-1] * current[idx]
            end
            current_idx += length(current)
        end
        return res
    end
end


ptrue = evaluate_poly(pushfirst!(collect(Iterators.flatten(all_coeffs)), 12.0))

@time true_poly(X);

const training_data = map(1:1000) do _
    x = 3 * randn(10)
    y = true_poly(x) + 2 * randn()
    (x, y)
end

function f(coefficients)
    p = evaluate_poly(coefficients)
    return 0.5 * sum(training_data) do (x, y)
        (p(x) - y)^2
    end
end

function grad!(storage::Array, p_coefficients::Array)
    ReverseDiff.gradient!(storage, f, p_coefficients)
end

function grad!(storage, p_coefficients)
    ReverseDiff.gradient!(convert(Array, storage), f, convert(Array, p_coefficients))
end


# function grad!(storage, p_coefficients)
#     p = evaluate_poly(p_coefficients)
#     d = sum(p(x) - y for (x, y) in training_data)
#     storage[1] = d
#     current = 1
#     current_idx = 2
#     for d in 1:max_degree
#         current = sum([xi * c for xi in xt for c in current] for (xt, _) in training_data)
#         @info size(current)
#         for idx in eachindex(current)
#             storage[current_idx+idx-1] = current[idx] * d
#         end
#         current_idx += length(current)
#     end
# end

# gradient descent
xgd = rand(length(flattened_coeff))
for _ in 1:2000
    grad!(gradient, xgd)
    xgd -= 0.05 * gradient
end

lmo = FrankWolfe.KSparseLMO(
    round(Int, length(flattened_coeff) / 4),
    1.1 * maximum(flattened_coeff)
)

x0 = FrankWolfe.compute_extreme_point(lmo, rand(length(flattened_coeff)))


k = 1000

# vanilla FW
@time x, v, primal, dual_gap, trajectoryFw = FrankWolfe.fw(
    f,
    grad!,
    lmo,
    collect(x0),
    max_iteration=k,
    line_search=FrankWolfe.shortstep,
    L=2,
    print_iter=k / 50,
    emphasis=FrankWolfe.blas,
    verbose=true,
    trajectory=true,
    gradient=gradient,
);

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
    trajectory=true,
    Ktolerance=0.95,
    goodstep_tolerance=0.95,
    weight_purge_threshold=1e-10,
)
