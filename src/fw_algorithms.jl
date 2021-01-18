
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

gradient_type(::SimpleFunction{VT, F, G}) = VT

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

