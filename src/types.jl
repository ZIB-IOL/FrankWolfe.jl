
"""
    ScaledHotVector{T}

Represents a vector of at most one value different from 0.
"""
struct ScaledHotVector{T} <: AbstractVector{T}
    active_val::T
    val_idx::Int
    len::Int
end

Base.size(v::ScaledHotVector) = (v.len,)

@inline function Base.getindex(v::ScaledHotVector{T}, idx::Integer) where {T}
    @boundscheck if !(1 ≤ idx ≤ length(v))
        throw(BoundsError(v, idx))
    end
    if v.val_idx != idx
        return zero(T)
    end
    return v.active_val
end

Base.sum(v::ScaledHotVector) = v.active_val

function LinearAlgebra.dot(v1::ScaledHotVector{<:Number}, v2::AbstractVector{<:Number})
    return conj(v1.active_val) * v2[v1.val_idx]
end

function LinearAlgebra.dot(
    v1::ScaledHotVector{<:Number},
    v2::SparseArrays.SparseVectorUnion{<:Number},
)
    return conj(v1.active_val) * v2[v1.val_idx]
end

LinearAlgebra.dot(v1::AbstractVector{<:Number}, v2::ScaledHotVector{<:Number}) = conj(dot(v2, v1))

LinearAlgebra.dot(v1::SparseArrays.SparseVectorUnion{<:Number}, v2::ScaledHotVector{<:Number}) =
    conj(dot(v2, v1))

function LinearAlgebra.dot(v1::ScaledHotVector{<:Number}, v2::ScaledHotVector{<:Number})
    if length(v1) != length(v2)
        throw(DimensionMismatch("v1 and v2 do not have matching sizes"))
    end
    return conj(v1.active_val) * v2.active_val * (v1.val_idx == v2.val_idx)
end

LinearAlgebra.norm(v::ScaledHotVector) = abs(v.active_val)

function Base.:*(v::ScaledHotVector, x::Number)
    return ScaledHotVector(v.active_val * x, v.val_idx, v.len)
end

Base.:*(x::Number, v::ScaledHotVector) = v * x

function Base.:+(x::ScaledHotVector, y::AbstractVector)
    if length(x) != length(y)
        throw(DimensionMismatch())
    end
    yc = Base.copymutable(y)
    @inbounds yc[x.val_idx] += x.active_val
    return yc
end

Base.:+(y::AbstractVector, x::ScaledHotVector) = x + y

function Base.:+(x::ScaledHotVector{T1}, y::ScaledHotVector{T2}) where {T1,T2}
    n = length(x)
    T = promote_type(T1, T2)
    if n != length(y)
        throw(DimensionMismatch())
    end
    res = spzeros(T, n)
    @inbounds res[x.val_idx] = x.active_val
    @inbounds res[y.val_idx] += y.active_val
    return res
end

Base.:-(x::ScaledHotVector{T}) where {T} = ScaledHotVector{T}(-x.active_val, x.val_idx, x.len)

Base.:-(x::AbstractVector, y::ScaledHotVector) = +(x, -y)
Base.:-(x::ScaledHotVector, y::AbstractVector) = +(x, -y)

Base.:-(x::ScaledHotVector, y::ScaledHotVector) = +(x, -y)

Base.similar(v::ScaledHotVector{T}) where {T} = spzeros(T, length(v))

function Base.convert(::Type{Vector{T}}, v::ScaledHotVector) where {T}
    vc = zeros(T, v.len)
    vc[v.val_idx] = v.active_val
    return vc
end

function Base.isequal(a::ScaledHotVector, b::ScaledHotVector)
    return a.len == b.len && a.val_idx == b.val_idx && isequal(a.active_val, b.active_val)
end

function Base.copyto!(dst::SparseArrays.SparseVector, src::ScaledHotVector)
    for idx in eachindex(src)
        dst[idx] = 0
    end
    SparseArrays.dropzeros!(dst)
    dst[src.val_idx] = src.active_val
    return dst
end

SparseArrays.issparse(::ScaledHotVector) = true

function active_set_update_scale!(x::IT, lambda, atom::ScaledHotVector) where {IT}
    x .*= (1 - lambda)
    x[atom.val_idx] += lambda * atom.active_val
    return x
end

"""
    RankOneMatrix{T, UT, VT}

Represents a rank-one matrix `R = u * vt'`.
Composes like a charm.
"""
struct RankOneMatrix{T,UT<:AbstractVector,VT<:AbstractVector} <: AbstractMatrix{T}
    u::UT
    v::VT
end

function RankOneMatrix(u::UT, v::VT) where {UT,VT}
    T = promote_type(eltype(u), eltype(v))
    return RankOneMatrix{T,UT,VT}(u, v)
end

# not checking indices
Base.@propagate_inbounds function Base.getindex(R::RankOneMatrix, i, j)
    @boundscheck (checkbounds(R.u, i); checkbounds(R.v, j))
    @inbounds R.u[i] * R.v[j]
end

Base.size(R::RankOneMatrix) = (length(R.u), length(R.v))
function Base.:*(R::RankOneMatrix, v::AbstractVector)
    temp = fast_dot(R.v, v)
    return R.u * temp
end

function Base.:*(R::RankOneMatrix, M::AbstractMatrix)
    temp = M' * R.v
    return RankOneMatrix(R.u, temp)
end

Base.:*(R::RankOneMatrix, D::LinearAlgebra.Diagonal) = RankOneMatrix(R.u, R.v .* D.diag)
Base.:*(R::RankOneMatrix, T::LinearAlgebra.AbstractTriangular) = RankOneMatrix(R.u, T' * R.v)

function Base.:*(R1::RankOneMatrix, R2::RankOneMatrix)
    # middle product
    temp = fast_dot(R1.v, R2.u)
    return RankOneMatrix(R1.u * temp, R2.v)
end

Base.Matrix(R::RankOneMatrix) = R.u * R.v'
Base.collect(R::RankOneMatrix) = Matrix(R)
Base.copymutable(R::RankOneMatrix) = Matrix(R)
Base.copy(R::RankOneMatrix) = RankOneMatrix(copy(R.u), copy(R.v))

function Base.convert(::Type{<:RankOneMatrix{T,Vector{T},Vector{T}}}, R::RankOneMatrix) where {T}
    return RankOneMatrix(convert(Vector{T}, R.u), convert(Vector{T}, R.v))
end

function LinearAlgebra.dot(
    R::RankOneMatrix{T1},
    S::SparseArrays.AbstractSparseMatrixCSC{T2},
) where {T1<:Real,T2<:Real}
    (m, n) = size(R)
    T = promote_type(T1, T2)
    if (m, n) != size(S)
        throw(DimensionMismatch("Size mismatch"))
    end
    s = zero(T)
    if m * n == 0
        return s
    end
    rows = SparseArrays.rowvals(S)
    vals = SparseArrays.nonzeros(S)
    @inbounds for j in 1:n
        for ridx in SparseArrays.nzrange(S, j)
            i = rows[ridx]
            v = vals[ridx]
            s += v * R.u[i] * R.v[j]
        end
    end
    return s
end

LinearAlgebra.dot(R::RankOneMatrix, M::Matrix) = dot(R.u, M, R.v)
LinearAlgebra.dot(M::Matrix, R::RankOneMatrix) = conj(dot(R, M))

Base.@propagate_inbounds function Base.:-(a::RankOneMatrix, b::RankOneMatrix)
    @boundscheck size(a) == size(b) || throw(DimensionMismatch())
    r = similar(a)
    @inbounds for j in 1:size(a, 2)
        for i in 1:size(a, 1)
            r[i, j] = a.u[i] * a.v[j] - b.u[i] * b.v[j]
        end
    end
    return r
end

Base.:-(x::RankOneMatrix) = RankOneMatrix(-x.u, x.v)

Base.:*(x::Number, m::RankOneMatrix) = RankOneMatrix(x * m.u, m.v)
Base.:*(m::RankOneMatrix, x::Number) = RankOneMatrix(x * m.u, m.v)

Base.@propagate_inbounds function Base.:+(a::RankOneMatrix, b::RankOneMatrix)
    @boundscheck size(a) == size(b) || throw(DimensionMismatch())
    r = similar(a)
    @inbounds for j in 1:size(a, 2)
        for i in 1:size(a, 1)
            r[i, j] = a.u[i] * a.v[j] + b.u[i] * b.v[j]
        end
    end
    return r
end

LinearAlgebra.norm(R::RankOneMatrix) = norm(R.u) * norm(R.v)

Base.@propagate_inbounds function Base.isequal(a::RankOneMatrix, b::RankOneMatrix)
    if size(a) != size(b)
        return false
    end
    if isequal(a.u, b.u) && isequal(a.v, b.v)
        return true
    end
    # needs to check actual values
    @inbounds for j in 1:size(a, 2)
        for i in 1:size(a, 1)
            if !isequal(a.u[i] * a.v[j], b.u[i] * b.v[j])
                return false
            end
        end
    end
    return true
end

Base.@propagate_inbounds function muladd_memory_mode(
    ::InplaceEmphasis,
    d::Matrix,
    x::Union{RankOneMatrix,Matrix},
    v::RankOneMatrix,
)
    @boundscheck size(d) == size(x) || throw(DimensionMismatch())
    @boundscheck size(d) == size(v) || throw(DimensionMismatch())
    m, n = size(d)
    @inbounds for j in 1:n
        for i in 1:m
            d[i, j] = x[i, j] - v[i, j]
        end
    end
    return d
end

Base.@propagate_inbounds function muladd_memory_mode(
    ::InplaceEmphasis,
    x::Matrix,
    gamma::Real,
    d::RankOneMatrix,
)
    @boundscheck size(d) == size(x) || throw(DimensionMismatch())
    m, n = size(x)
    @inbounds for j in 1:n
        for i in 1:m
            x[i, j] -= gamma * d[i, j]
        end
    end
    return x
end

"""
    SubspaceVector{HasMultiplicities, T}

Companion structure of `SubspaceLMO` containing three fields:
- `data` is the full structure to be deflated,
- `vec` is a vector in the reduced subspace in which computations are performed,
- `mul` is only used to compute scalar products when `HasMultiplicities = true`,
which should be avoided (when possible) for performance reasons.
"""
struct SubspaceVector{HasMultiplicities,T,DT,V<:AbstractVector{T}} <: AbstractVector{T}
    data::DT # full array to be symmetrised, will generally fall out of sync wrt vec
    vec::V # vector representing the array
    mul::Vector{T} # only used for scalar products
end

function SubspaceVector(data::DT, vec::V) where {DT,V<:AbstractVector{T}} where {T}
    return SubspaceVector{false,T,DT,V}(data, vec, T[])
end

function SubspaceVector(data::DT, vec::V, mul::Vector) where {DT,V<:AbstractVector{T}} where {T}
    return SubspaceVector{true,T,DT,V}(data, vec, convert(Vector{T}, mul))
end

Base.@propagate_inbounds function Base.getindex(A::SubspaceVector, i)
    @boundscheck checkbounds(A.vec, i)
    return @inbounds getindex(A.vec, i)
end

Base.@propagate_inbounds function Base.setindex!(A::SubspaceVector, x, i)
    @boundscheck checkbounds(A.vec, i)
    return @inbounds setindex!(A.vec, x, i)
end

Base.size(A::SubspaceVector) = size(A.vec)
Base.eltype(A::SubspaceVector) = eltype(A.vec)
Base.similar(A::SubspaceVector{true}) = SubspaceVector(similar(A.data), similar(A.vec), A.mul)
Base.similar(A::SubspaceVector{false}) = SubspaceVector(similar(A.data), similar(A.vec))
Base.similar(A::SubspaceVector{true}, ::Type{T}) where {T} =
    SubspaceVector(similar(A.data, T), similar(A.vec, T), convert(Vector{T}, A.mul))
Base.similar(A::SubspaceVector{false}, ::Type{T}) where {T} =
    SubspaceVector(similar(A.data, T), similar(A.vec, T))
Base.collect(A::SubspaceVector{true}) = SubspaceVector(collect(A.data), collect(A.vec), A.mul)
Base.collect(A::SubspaceVector{false}) = SubspaceVector(collect(A.data), collect(A.vec))
Base.copyto!(dest::SubspaceVector, src::SubspaceVector) = copyto!(dest.vec, src.vec)
Base.:*(scalar::Real, A::SubspaceVector{true}) = SubspaceVector(A.data, scalar * A.vec, A.mul)
Base.:*(scalar::Real, A::SubspaceVector{false}) = SubspaceVector(A.data, scalar * A.vec)
Base.:*(A::SubspaceVector, scalar::Real) = scalar * A
Base.:/(A::SubspaceVector, scalar::Real) = inv(scalar) * A
Base.:+(A1::SubspaceVector{true,T}, A2::SubspaceVector{true,T}) where {T} =
    SubspaceVector(A1.data, A1.vec + A2.vec, A1.mul)
Base.:+(A1::SubspaceVector{false,T}, A2::SubspaceVector{false,T}) where {T} =
    SubspaceVector(A1.data, A1.vec + A2.vec)
Base.:-(A1::SubspaceVector{true,T}, A2::SubspaceVector{true,T}) where {T} =
    SubspaceVector(A1.data, A1.vec - A2.vec, A1.mul)
Base.:-(A1::SubspaceVector{false,T}, A2::SubspaceVector{false,T}) where {T} =
    SubspaceVector(A1.data, A1.vec - A2.vec)
Base.:-(A::SubspaceVector{true,T}) where {T} = SubspaceVector(A.data, -A.vec, A.mul)
Base.:-(A::SubspaceVector{false,T}) where {T} = SubspaceVector(A.data, -A.vec)

LinearAlgebra.dot(A1::SubspaceVector{true}, A2::SubspaceVector{true}) =
    dot(A1.vec, Diagonal(A1.mul), A2.vec)
LinearAlgebra.dot(A1::SubspaceVector{false}, A2::SubspaceVector{false}) = dot(A1.vec, A2.vec)
LinearAlgebra.norm(A::SubspaceVector) = sqrt(dot(A, A))

Base.@propagate_inbounds Base.isequal(A1::SubspaceVector, A2::SubspaceVector) =
    isequal(A1.vec, A2.vec)

"""
Given an array `array`, `NegatingArray` represents `-1 * array` lazily.
"""
struct NegatingArray{T,N,AT<:AbstractArray{T,N}} <: AbstractArray{T,N}
    array::AT
    function NegatingArray(array::AT) where {T,N,AT<:AbstractArray{T,N}}
        return new{T,N,AT}(array)
    end
end

Base.size(a::NegatingArray) = Base.size(a.array)
Base.getindex(a::NegatingArray, idxs...) = -Base.getindex(a.array, idxs...)

LinearAlgebra.dot(a1::NegatingArray{<:Number}, a2::NegatingArray{<:Number}) =
    dot(a1.array, a2.array)
LinearAlgebra.dot(a1::NegatingArray{<:Number,N}, a2::AbstractArray{<:Number,N}) where {N} =
    -dot(a1.array, a2)
LinearAlgebra.dot(a1::AbstractArray{<:Number,N}, a2::NegatingArray{<:Number,N}) where {N} =
    -dot(a1, a2.array)

# removing method ambiguities
LinearAlgebra.dot(a1::LinearAlgebra.Diagonal, a2::NegatingArray{<:Number}) = -dot(a1, a2.array)
LinearAlgebra.dot(a1::NegatingArray{<:Number}, a2::SparseArrays.SparseVectorUnion) =
    -dot(a1.array, a2)
LinearAlgebra.dot(a1::SparseArrays.SparseVectorUnion, a2::NegatingArray{<:Number}) =
    -dot(a1, a2.array)

Base.sum(a::NegatingArray) = -sum(a.array)
LinearAlgebra.dot(v1::NegatingArray{<:Number,1}, v2::ScaledHotVector{<:Number}) = -dot(v1.array, v2)
LinearAlgebra.dot(v1::ScaledHotVector{<:Number}, v2::NegatingArray{<:Number,1}) = -dot(v1, v2.array)

LinearAlgebra.dot(a::NegatingArray{<:Number,2}, d::LinearAlgebra.Diagonal) = -dot(a.array, d)
LinearAlgebra.dot(d::LinearAlgebra.Diagonal, a::NegatingArray{<:Number,2}) = -dot(d, a.array)
