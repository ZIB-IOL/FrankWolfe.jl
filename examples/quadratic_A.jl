using FrankWolfe
using Random
using LinearAlgebra
Random.seed!(0)

n = 5 # number of dimensions
p = 10^3 # number of points
k = 10^4 # number of iterations
T = Float64

function simple_reg_loss(θ, data_point)
    (xi, yi) = data_point
    (a, b) = (θ[1:end-1], θ[end])
    pred = a ⋅ xi + b
    return (pred - yi)^2 / 2
end

function ∇simple_reg_loss(storage, θ, data_point)
    (xi, yi) = data_point
    (a, b) = (θ[1:end-1], θ[end])
    pred = a ⋅ xi + b
    @. storage[1:end-1] += xi * (pred - yi)
    storage[end] += pred - yi
    return storage
end

xs = [10randn(T, n) for _ in 1:p]
bias = 4
params_perfect = [1:n; bias]

# similar example with noisy data, Gaussian noise around the linear estimate
data_noisy = [(x, x ⋅ (1:n) + bias + 0.5 * randn(T)) for x in xs]

f(x) = sum(simple_reg_loss(x, data_point) for data_point in data_noisy)

function gradf(storage, x)
    storage .= 0
    for dp in data_noisy
        ∇simple_reg_loss(storage, x, dp)
    end
end

lmo = FrankWolfe.LpNormLMO{T, 2}(1.05 * norm(params_perfect))

x0 = FrankWolfe.compute_extreme_point(lmo, zeros(T, n+1))

# standard active set
# active_set = FrankWolfe.ActiveSet([(1.0, x0)])

# specialized active set, automatically detecting the parameters A and b of the quadratic function f
active_set = FrankWolfe.ActiveSetQuadraticCachedProducts([(one(T), x0)], gradf)

@time res = FrankWolfe.blended_pairwise_conditional_gradient(
#  @time res = FrankWolfe.away_frank_wolfe(
    f,
    gradf,
    lmo,
    active_set;
    verbose=true,
    lazy=true,
    line_search=FrankWolfe.Adaptive(L_est=10.0, relaxed_smoothness=true),
    max_iteration=k,
    print_iter=k / 10,
    trajectory=true,
)

println()
