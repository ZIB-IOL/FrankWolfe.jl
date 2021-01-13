
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

UnitSimplexOracle{T}() = UnitSimplexOracle{T}(one(T))

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

ProbabilitySimplexOracle{T}() = ProbabilitySimplexOracle{T}(one(T))

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

"""
    MaybeHotVector{T}

Represents a vector of at most one value different from 0.
"""
struct MaybeHotVector{T} <: AbstractVector{T}
    active_val::T
    val_idx::Int
    len::Int
end

Base.size(v::MaybeHotVector) = (v.len, )

@inline function Base.getindex(v::MaybeHotVector{T}, idx) where {T}
    @boundscheck if !( 1 ≤ idx ≤ length(v))
        throw(BoundsError(v, idx))
    end
    if v.val_idx != idx
        return zero(T)
    end
    return v.active_val
end

Base.sum(v::MaybeHotVector) = v.active_val

function LinearAlgebra.dot(v1::MaybeHotVector, v2::AbstractVector)
    return v1.active_val * v2[v1.val_idx]
end

LinearAlgebra.dot(v1, v2::MaybeHotVector) = LinearAlgebra.dot(v2, v1)
