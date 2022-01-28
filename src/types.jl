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

function LinearAlgebra.dot(v1::ScaledHotVector{<:Number}, v2::SparseArrays.SparseVector{<:Number})
    return conj(v1.active_val) * v2[v1.val_idx]
end

LinearAlgebra.dot(v1::AbstractVector{<:Number}, v2::ScaledHotVector{<:Number}) = conj(dot(v2, v1))

LinearAlgebra.dot(v1::SparseArrays.SparseVector{<:Number}, v2::ScaledHotVector{<:Number}) = conj(dot(v2, v1))

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
    temp = R.v' * M
    return RankOneMatrix(u, temp')
end

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
