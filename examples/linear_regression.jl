"""
Example of a L1-constrained linearized regression
using the stochastic version of Frank Wolfe.
"""

using FrankWolfe
using Random
using LinearAlgebra

import FrankWolfe: compute_gradient, compute_value, compute_value_gradient

using Test

# user-provided loss function and gradient

function simple_reg_loss(θ, data_point)
    (xi, yi) = data_point
    (a, b) = (θ[1:end-1], θ[end])
    pred = a ⋅ xi + b
    return (pred - yi)^2 / 2
end

function ∇simple_reg_loss(θ, data_point)
    (xi, yi) = data_point
    (a, b) = (θ[1:end-1], θ[end])
    pred = a ⋅ xi + b
    storage = [xi * (pred - yi); pred - yi]
    return storage
end

function ∇simple_reg_loss(storage, θ, data_point)
    (xi, yi) = data_point
    (a, b) = (θ[1:end-1], θ[end])
    pred = a ⋅ xi + b
    @. storage[1:end-1] += xi * (pred - yi)
    storage[end] += pred - yi
    return storage
end

xs = [10 * randn(5) for i in 1:20000]
params = rand(6) .- 1 # start params in (-1,0)
bias = 4π
params_perfect = [1:5; bias]

data_perfect = [(x, x ⋅ (1:5) + bias) for x in xs]
f_stoch = FrankWolfe.StochasticObjective(simple_reg_loss, ∇simple_reg_loss, data_perfect)

@test compute_value(f_stoch, params) > compute_value(f_stoch, params_perfect)

# Vanilla Stochastic Gradient Descent with reshuffling
storage = 0 * similar(params)
for idx in 1:1000
    for data_point in shuffle!(data_perfect)
        params .-= 0.05 * ∇simple_reg_loss(params, data_point) / length(data_perfect)
    end
end

@test norm(params - params_perfect) <= 1e-6

# similar example with noisy data, Gaussian noise around the linear estimate
data_noisy = [(x, x ⋅ (1:5) + bias + 0.5 * randn()) for x in xs]
f_stoch_noisy = FrankWolfe.StochasticObjective(simple_reg_loss, ∇simple_reg_loss, data_noisy)

params = rand(6) .- 1 # start params in (-1,0)

# test that the true parameters yield a good error
@test norm(compute_gradient(f_stoch_noisy, params_perfect)) <= length(data_noisy) * 0.05

# test that gradient at true parameters has lower norm than at randomly initialized ones
@test norm(compute_gradient(f_stoch_noisy, params_perfect)) <
      norm(compute_gradient(f_stoch_noisy, params))

# test that error at true parameters is lower than at randomly initialized ones
@test compute_value(f_stoch_noisy, params) > compute_value(f_stoch_noisy, params_perfect)

# Vanilla Stochastic Gradient Descent with reshuffling
for idx in 1:1000
    for data_point in shuffle!(data_perfect)
        params .-= 0.05 * ∇simple_reg_loss(params, data_point) / length(data_perfect)
    end
end

# test that SG converged towards true parameters 
@test norm(params - params_perfect) <= 1e-6

#####
# Stochastic Frank Wolfe version
# We constrain the argument in the L2-norm ball with a large-enough radius

lmo = FrankWolfe.LpNormLMO{2}(1.05 * norm(params_perfect))

params = rand(6) .- 1 # start params in (-1,0)

k = 10000

@time x, v, primal, dual_gap, trajectoryS = FrankWolfe.stochastic_frank_wolfe(
    f_stoch_noisy,
    lmo,
    params,
    verbose=true,
    rng=Random.GLOBAL_RNG,
    line_search=FrankWolfe.nonconvex,
    max_iteration=k,
    print_iter=k / 10,
    batch_size=length(f_stoch_noisy.xs) ÷ 100 + 1,
    trajectory=true,
)

@time x, v, primal, dual_gap, trajectory09 = FrankWolfe.stochastic_frank_wolfe(
    f_stoch_noisy,
    lmo,
    params,
    momentum=0.9,
    verbose=true,
    rng=Random.GLOBAL_RNG,
    line_search=FrankWolfe.nonconvex,
    max_iteration=k,
    print_iter=k / 10,
    batch_size=length(f_stoch_noisy.xs) ÷ 100 + 1,
    trajectory=true,
)

@time x, v, primal, dual_gap, trajectory099 = FrankWolfe.stochastic_frank_wolfe(
    f_stoch_noisy,
    lmo,
    params,
    momentum=0.99,
    verbose=true,
    rng=Random.GLOBAL_RNG,
    line_search=FrankWolfe.nonconvex,
    max_iteration=k,
    print_iter=k / 10,
    batch_size=length(f_stoch_noisy.xs) ÷ 100 + 1,
    trajectory=true,
)


const ff = x -> compute_value(f_stoch_noisy, x, full_evaluation=true)
function gradf(storage, x)
    storage .= 0
    for dp in data_noisy
        ∇simple_reg_loss(storage, x, dp)
    end
end

@time x, v, primal, dual_gap, trajectory = FrankWolfe.frank_wolfe(
    ff,
    gradf,
    lmo,
    L=10,
    params,
    verbose=true,
    line_search=FrankWolfe.adaptive,
    max_iteration=k,
    print_iter=k / 10,
    trajectory=true,
)

data = [trajectory, trajectoryS, trajectory09, trajectory099]
label = ["exact", "stochastic", "stochM 0.9", "stochM 0.99"]

FrankWolfe.plot_trajectories(data, label)
