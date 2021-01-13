
"""
    UnitSimplexOracle(right_side)

Represents the scaled unit simplex:
```
C = {x ∈ R^n_+, ∑x ≤ right_side}
```
"""
struct UnitSimplexOracle{T} <: LinearMinimizationOracle
    right_side::T
end

UnitSimplexOracle{T}() where {T} = UnitSimplexOracle{T}(one(T))

"""
LMO for scaled unit simplex.
Returns either vector of zeros or vector with one active value equal to RHS if
there exists an improving direction.
"""
function compute_extreme_point(lmo::UnitSimplexOracle{T}, direction) where {T}
    idx = argmin(direction)
    if direction[idx] < 0
        return MaybeHotVector(lmo.right_side, idx, length(direction))
    end
    return MaybeHotVector(zero(T), idx, length(direction))
end

function unitSimplexLMO(grad;r=1)
    n = length(grad)
    v = zeros(n)
    aux = argmin(grad)
    if grad[aux] < 0.0
        v[aux] = 1.0
    end
    return v*r
end

"""
    ProbabilitySimplexOracle(right_side)

Represents the scaled probability simplex:
```
C = {x ∈ R^n_+, ∑x = right_side}
```
"""
struct ProbabilitySimplexOracle{T} <: LinearMinimizationOracle
    right_side::T
end

ProbabilitySimplexOracle{T}() where {T} = ProbabilitySimplexOracle{T}(one(T))

"""
LMO for scaled probability simplex.
Returns a vector with one active value equal to RHS in the
most improving (or least degrading) direction.
"""
function compute_extreme_point(lmo::ProbabilitySimplexOracle{T}, direction) where {T}
    idx = argmin(direction)
    return MaybeHotVector(lmo.right_side, idx, length(direction))
end


# simple probabilitySimplexLMO
# TODO:
# - not optimized

function probabilitySimplexLMO(grad;r=1)
    n = length(grad)
    v = zeros(n)
    aux = argmin(grad)
    v[aux] = 1.0
    return v*r
end
