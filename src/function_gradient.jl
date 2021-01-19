
"""
    ObjectiveFunction{VT}

Represents an objective function optimized by algorithms.
Subtypes of `ObjectiveFunction` must implement at least
* `compute_value(::ObjectiveFunction, x)` for primal value evaluation
* `compute_gradient(::ObjectiveFunction, x)` for gradient evaluation.
and optionally `compute_value_gradient(::ObjectiveFunction, x)` returning the (primal, gradient) pair.
"""
abstract type ObjectiveFunction end

function compute_value end

function compute_gradient end

"""
    compute_value_gradient(f::ObjectiveFunction, x; kwargs)

Computes in one call the pair `(function_value, function_grad)` evaluated at `x`.
By default, calls `compute_value` and `compute_gradient` with keyword `kwargs`
passed to both.
"""
compute_value_gradient(f::ObjectiveFunction, x; kwargs...) =
    (compute_value(f, x; kwargs...), compute_gradient(f, x; kwargs...))

struct SimpleFunctionObjective{F,G} <: ObjectiveFunction
    f::F
    grad::G
end

compute_value(f::SimpleFunctionObjective, x) = f.f(x)
compute_gradient(f::SimpleFunctionObjective, x) = f.grad(x)

"""
    StochasticObjective{F, G, XT}(f::F, grad::G, xs::XT)

Represents an objective function evaluated with stochastic gradient.
`f(θ, x)` evaluates the loss for data point `x` and parameter `θ`.
`grad(θ, x)` evaluates the loss gradient with respect to data point `x` at parameter `θ`.
`xs` must be an indexable iterable (`Vector{Vector{Float64}}` for instance).
Functions using a `StochasticObjective` have optional keyword arguments `rng`, `batch_size`
and `full_evaluation` controlling whether the function should be evaluated over all data points.
"""
struct StochasticObjective{F,G,XT} <: ObjectiveFunction
    f::F
    grad::G
    xs::XT
end

function compute_value(
    f::StochasticObjective,
    θ;
    batch_size::Integer=length(f.xs),
    rng=Random.GLOBAL_RNG,
    full_evaluation=false,
)
    rand_indices = if full_evaluation
        eachindex(f.xs)
    else
        rand(rng, eachindex(f.xs), batch_size)
    end
    return sum(f.f(θ, f.xs[idx]) for idx in rand_indices)
end

function compute_gradient(
    f::StochasticObjective,
    θ;
    batch_size::Integer=length(f.xs) ÷ 10 + 1,
    rng=Random.GLOBAL_RNG,
    full_evaluation=false,
)
    rand_indices = if full_evaluation
        eachindex(f.xs)
    else
        rand(rng, eachindex(f.xs), batch_size)
    end
    return sum(f.grad(θ, f.xs[idx]) for idx in rand_indices)
end

function compute_value_gradient(
    f::StochasticObjective,
    θ;
    batch_size::Integer=length(f.xs) ÷ 10 + 1,
    rng=Random.GLOBAL_RNG,
    full_evaluation=false,
)
    rand_indices = if full_evaluation
        eachindex(f.xs)
    else
        rand(rng, eachindex(f.xs), batch_size)
    end
    # map operation, for each index, computes value and gradient
    function map_op(idx)
        @inbounds x = f.xs[idx]
        return (f.f(θ, x), f.grad(θ, x))
    end
    # reduce: take partial value and gradient, adds value and gradient wrt new point
    function reduce_op(left_tup, right_tup)
        (f_val, g_val) = left_tup
        (f_new, g_new) = right_tup
        return (f_val + f_new, g_val + g_new)
    end

    return mapfoldr(map_op, reduce_op, rand_indices)
end
