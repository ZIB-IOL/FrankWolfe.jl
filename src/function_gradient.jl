
abstract type ObjectiveFunction
end

function compute_value end

function compute_gradient end

function compute_value_gradient end

function gradient_type end

struct SimpleFunction{VT, F, G} <: ObjectiveFunction
    f::F
    grad::G
end

SimpleFunction(f::F, grad::G) where {F, G} = SimpleFunction{Vector{Float64}, F, G}(f, g)

compute_value(f::SimpleFunction, x) = f.f(x)
compute_gradient(f::SimpleFunction, x) = f.grad(x)
compute_value_gradient(f::SimpleFunction, x) = (f.f(x), f.grad(x))

gradient_type(::SimpleFunction{VT}) where {VT} = VT

"""
    StochasticGradientObjective{VT, F, G, XT}(f::F, grad::G, xs::XT)

Represents an objective function evaluated with stochastic gradient.
`f(θ, x)` evaluates the loss for data point `x` and parameter `θ`.
`grad(θ, x)` evaluates the loss gradient with respect to data point `x` at parameter `θ`.
`xs` must be an indexable iterable (by default a `Vector{Vector{Float64}}`).
`VT` is the type returned by `grad(θ, x)`.
"""
struct StochasticObjective{VT, F, G, XT} <: ObjectiveFunction
    f::F
    grad::G
    xs::XT
end

function StochasticObjective{VT}(f::F, grad::G, xs::XT) where {VT, F, G, XT}
    return StochasticObjective{VT, F, G, XT}(f::F, grad::G, xs::XT)
end

function compute_value(f::StochasticObjective, θ; batch_size::Integer=length(f.xs), rng=Random.GLOBAL_RNG)
    rand_indices = rand(rng, eachindex(f.xs), batch_size)
    return sum(f.f(θ, f.xs[idx]) for idx in rand_indices)
end

function compute_gradient(f::StochasticObjective, θ; batch_size::Integer=length(f.xs) ÷ 10, rng=Random.GLOBAL_RNG)
    rand_indices = rand(rng, eachindex(f.xs), batch_size)
    return sum(f.grad(θ, f.xs[idx]) for idx in rand_indices)
end

function compute_value_gradient(f::StochasticObjective, θ; batch_size::Integer=length(f.xs) ÷ 10, rng=Random.GLOBAL_RNG)
    rand_indices = rand(rng, eachindex(f.xs), batch_size)
    # reduce: take partial value and gradient, and new index, add value and gradient wrt new point
    function reduce_op(partial_values, idx)
        (f_val, g_val) = partial_values
        x = @inbounds f.xs[idx]
        (f_val + f.f(θ, x), g_val + f.grad(θ, x))
    end
    mapreduce(
        idx -> (f.f(θ, f.xs[idx]), f.grad(θ, f.xs[idx])), # map operation
        reduce_op,
        rand_indices,
    )
end

gradient_type(::StochasticObjective{VT}) where {VT} = VT
