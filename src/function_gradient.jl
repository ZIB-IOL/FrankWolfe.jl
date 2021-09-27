
"""
    ObjectiveFunction

Represents an objective function optimized by algorithms.
Subtypes of `ObjectiveFunction` must implement at least
* `compute_value(::ObjectiveFunction, x)` for primal value evaluation
* `compute_gradient(::ObjectiveFunction, x)` for gradient evaluation.
and optionally `compute_value_gradient(::ObjectiveFunction, x)` returning the (primal, gradient) pair.
`compute_gradient` may always use the same storage and return a reference to it.
"""
abstract type ObjectiveFunction end

"""
    compute_value(f::ObjectiveFunction, x; [kwargs...])

Computes the objective `f` at `x`.
"""
function compute_value end

"""
    compute_gradient(f::ObjectiveFunction, x; [kwargs...])

Computes the gradient of `f` at `x`. May return a reference to an internal storage.
"""
function compute_gradient end

"""
    compute_value_gradient(f::ObjectiveFunction, x; [kwargs...])

Computes in one call the pair `(value, gradient)` evaluated at `x`.
By default, calls `compute_value` and `compute_gradient` with keywords `kwargs`
passed down to both.
"""
compute_value_gradient(f::ObjectiveFunction, x; kwargs...) =
    (compute_value(f, x; kwargs...), compute_gradient(f, x; kwargs...))

"""
    SimpleFunctionObjective{F,G,S}

An objective function built from separate primal objective `f(x)` and
in-place gradient function `grad!(storage, x)`.
It keeps an internal storage of type `s` used to evaluate the gradient in-place.
"""
struct SimpleFunctionObjective{F,G,S} <: ObjectiveFunction
    f::F
    grad!::G
    storage::S
end

compute_value(f::SimpleFunctionObjective, x) = f.f(x)
function compute_gradient(f::SimpleFunctionObjective, x)
    f.grad!(f.storage, x)
    return f.storage
end

"""
    StochasticObjective{F, G, XT, S}(f::F, grad!::G, xs::XT, storage::S)

Represents a composite function evaluated with stochastic gradient.
`f(θ, x)` evaluates the loss for data point `x` and parameter `θ`.
`grad!(storage, θ, x)` adds to storage the partial gradient with respect to data point `x` at parameter `θ`.
`xs` must be an indexable iterable (`Vector{Vector{Float64}}` for instance).
Functions using a `StochasticObjective` have optional keyword arguments `rng`, `batch_size`
and `full_evaluation` controlling whether the function should be evaluated over all data points.

Note: `grad!` must **not** reset the storage to 0 before adding to it.
"""
struct StochasticObjective{F,G,XT,S} <: ObjectiveFunction
    f::F
    grad!::G
    xs::XT
    storage::S
end

function compute_value(
    f::StochasticObjective,
    θ;
    batch_size::Integer=length(f.xs),
    rng=Random.GLOBAL_RNG,
    full_evaluation=false,
)
    (batch_size, rand_indices) = if full_evaluation
        (length(f.xs), eachindex(f.xs))
    else
        (batch_size, rand(rng, eachindex(f.xs), batch_size))
    end
    return sum(f.f(θ, f.xs[idx]) for idx in rand_indices) / batch_size
end

function compute_gradient(
    f::StochasticObjective,
    θ;
    batch_size::Integer=length(f.xs) ÷ 10 + 1,
    rng=Random.GLOBAL_RNG,
    full_evaluation=false,
)
    (batch_size, rand_indices) = if full_evaluation
        (length(f.xs), eachindex(f.xs))
    else
        (batch_size, rand(rng, eachindex(f.xs), batch_size))
    end
    f.storage .= 0
    for idx in rand_indices
        f.grad!(f.storage, θ, f.xs[idx])
    end
    f.storage ./= batch_size
    return f.storage
end

function compute_value_gradient(
    f::StochasticObjective,
    θ;
    batch_size::Integer=length(f.xs) ÷ 10 + 1,
    rng=Random.GLOBAL_RNG,
    full_evaluation=false,
)
    (batch_size, rand_indices) = if full_evaluation
        (length(f.xs), eachindex(f.xs))
    else
        (batch_size, rand(rng, eachindex(f.xs), batch_size))
    end
    # map operation, for each index, computes value and gradient
    f_val = zero(eltype(θ))
    f.storage .= 0
    for idx in rand_indices
        @inbounds x = f.xs[idx]
        f_val += f.f(θ, x)
        f.grad!(f.storage, θ, x)
    end
    f.storage ./= batch_size
    f_val /= batch_size
    return (f_val, f.storage)
end
