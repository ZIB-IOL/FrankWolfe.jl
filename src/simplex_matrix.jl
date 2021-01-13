
"""
    SimplexMatrix{T}

Represents the simplex constraint matrix:
```
S x = 1
```
Has dimension `1×dim`
"""
struct SimplexMatrix{T} <: AbstractMatrix{T}
    dim::Int
end

Base.size(s::SimplexMatrix) = (1, s.dim)

@inline function Base.getindex(s::SimplexMatrix{T}, i::Integer, j::Integer) where {T}
    @boundscheck if i != 1 || !( 1 ≤ j ≤ s.dim )
        throw(BoundsError(s, (i, j)))
    end
    return one(T)
end

Base.getindex(s::SimplexMatrix, idx::Integer) = s[1,idx]

"""
    *(S::SimplexMatrix{T1}, v::AbstractVector{T2})

Product of the simplex matrix with a vector.
Returns a 1-element vector `[sum(v)]` of appropriate type.
"""
function Base.:*(S::SimplexMatrix{T1}, v::AbstractVector{T2}) where {T1, T2}
    if length(v) != size(S, 2)
        throw(DimensionMismatch("Vector length $(length(v)), simplex matrix dimension $(size(S))"))
    end
    T = promote_type(T1, T2)
    return T[sum(v)]
end

function Base.:*(S::SimplexMatrix{T1}, M::AbstractMatrix{T2}) where {T1, T2}
    if size(M, 1) != size(S, 2)
        throw(DimensionMismatch("$(size(S)), $(size(M))"))
    end
    T = promote_type(T1, T2)
    res = Matrix{T}(undef, 1, size(M, 2))
    @inbounds for j in 1:size(M, 2)
        res[1, j] = sum(view(M, :, j))
    end
    return res
end

function Base.:*(M::AbstractMatrix{T2}, S::SimplexMatrix{T1}) where {T1, T2}
    if size(M, 2) != 1
        throw(DimensionMismatch("$(size(M)), $(size(S))"))
    end
    T = promote_type(T1, T2)
    res = Matrix{T}(undef, size(M, 1), size(S, 2))
    @inbounds for j in 1:size(M, 1)
        for i in 1:size(S, 2)
            res[j, i] = M[j, 1]
        end
    end
    return res
end

function Base.convert(::Type{SimplexMatrix{T}}, s::SimplexMatrix) where {T}
    SimplexMatrix{T}(s.dim)
end
