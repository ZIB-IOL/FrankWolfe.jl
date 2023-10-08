mutable struct BlockVector{T, MT <: AbstractArray{T}, ST <: Tuple} <: AbstractVector{T}
    blocks::Vector{MT}
    block_sizes::Vector{ST}
    tot_size::Int
end

function BlockVector(arrays::AbstractVector{MT}) where {T, MT <: AbstractArray{T}}
    block_sizes = size.(arrays)
    tot_size = sum(prod, block_sizes)
    return BlockVector(arrays, block_sizes, tot_size)
end

Base.size(arr::BlockVector) = (arr.tot_size, )

# returns the corresponding (block_index, index_in_block) for a given flattened index (for the whole block variable)
function _matching_index_index(arr::BlockVector, idx::Integer)
    if idx < 1 || idx > length(arr)
        throw(BoundsError(arr, idx))
    end
    first_idx = 1
    for block_idx in eachindex(arr.block_sizes)
        next_first = first_idx + prod(arr.block_sizes[block_idx])
        if next_first <= idx
            # continue to next block
            first_idx = next_first
        else
            # index is here
            index_in_block = idx - first_idx + 1
            return (block_idx, index_in_block)
        end
    end
    error("unreachable $idx")
end

function Base.getindex(arr::BlockVector, idx::Integer)
    (midx, idx_inner) = _matching_index_index(arr, idx)
    return arr.blocks[midx][idx_inner]
end

function Base.setindex!(arr::BlockVector, v, idx::Integer)
    (midx, idx_inner) = _matching_index_index(arr, idx)
    arr.blocks[midx][idx_inner] = v
    return arr.blocks[midx][idx_inner]
end


function Base.copyto!(dest::BlockVector, src::BlockVector)
    dest.tot_size = src.tot_size
    for midx in eachindex(dest.blocks)
        dest.blocks[midx] = copy(src.blocks[midx])
    end
    dest.block_sizes = copy(src.block_sizes)
    return dest
end

function Base.similar(src::BlockVector{T1, MT}, ::Type{T}) where {T1, MT, T}
    blocks = [similar(src.blocks[i], T) for i in eachindex(src.blocks)]
    return BlockVector(
        blocks,
        src.block_sizes,
        src.tot_size,
    )
end

Base.similar(src::BlockVector{T, MT}) where {T, MT} = similar(src, T)

function Base.convert(::Type{BlockVector{T, MT}}, bmv::BlockVector) where {T, MT}
    cblocks = convert.(MT, bmv.blocks)
    return BlockVector(
        cblocks,
        copy(bmv.block_sizes),
        bmv.tot_size,
    )
end

function Base.:+(v1::BlockVector, v2::BlockVector)
    if size(v1) != size(v2) || length(v1.block_sizes) != length(v2.block_sizes)
        throw(DimensionMismatch("$(length(v1)) != $(length(v2))"))
    end
    for i in eachindex(v1.block_sizes)
        if v1.block_sizes[i] != v2.block_sizes[i]
            throw(DimensionMismatch("$i-th block: $(v1.block_sizes[i]) != $(v2.block_sizes[i])"))
        end
    end
    return BlockVector(
        v1.blocks .+ v2.blocks,
        copy(v1.block_sizes),
        v1.tot_size,
    )
end

Base.:-(v::BlockVector) = BlockVector(
    [-b for b in v.blocks],
    v.block_sizes,
    v.tot_size,
)

function Base.:-(v1::BlockVector, v2::BlockVector)
    return v1 + (-v2)
end

function Base.:*(s::Number, v::BlockVector)
    return BlockVector(
        s .* v.blocks,
        copy(v.block_sizes),
        v.tot_size,
    )
end

Base.:*(v::BlockVector, s::Number) = s * v

function LinearAlgebra.dot(v1::BlockVector{T1}, v2::BlockVector{T2}) where {T1, T2}
    if size(v1) != size(v2) || length(v1.block_sizes) != length(v2.block_sizes)
        throw(DimensionMismatch("$(length(v1)) != $(length(v2))"))
    end
    T = promote_type(T1, T2)
    d = zero(T)
    @inbounds for i in eachindex(v1.block_sizes)
        if v1.block_sizes[i] != v2.block_sizes[i]
            throw(DimensionMismatch("$i-th block: $(v1.block_sizes[i]) != $(v2.block_sizes[i])"))
        end
        d += dot(v1.blocks[i], v2.blocks[i])
    end
    return d
end

LinearAlgebra.norm(v::BlockVector) = sqrt(dot(v, v))

function Base.isequal(v1::BlockVector, v2::BlockVector)
    if v1 === v2
        return true
    end
    if v1.tot_size != v2.tot_size || v1.block_sizes != v2.block_sizes
        return false
    end
    for bidx in eachindex(v1.blocks)
        if !isequal(v1.blocks[bidx], v2.blocks[bidx])
            return false
        end
    end
    return true
end

"""
    ProductLMO(lmos)

Linear minimization oracle over the Cartesian product of multiple LMOs.
"""
struct ProductLMO{N, LT <: Union{NTuple{N, FrankWolfe.LinearMinimizationOracle}, AbstractVector{<: FrankWolfe.LinearMinimizationOracle}}} <: FrankWolfe.LinearMinimizationOracle
    lmos::LT
end

function ProductLMO(lmos::Vector{LMO}) where {LMO <: FrankWolfe.LinearMinimizationOracle}
    return ProductLMO{1, Vector{LMO}}(lmos)
end

function ProductLMO(lmos::NT) where {N, LMO <: FrankWolfe.LinearMinimizationOracle, NT <: NTuple{N, LMO}}
    return ProductLMO{N, NT}(lmos)
end

function FrankWolfe.compute_extreme_point(lmo::ProductLMO, direction::BlockVector; kwargs...)
    @assert length(direction.blocks) == length(lmo.lmos)
    blocks = [FrankWolfe.compute_extreme_point(lmo.lmos[idx], direction.blocks[idx]; kwargs...) for idx in eachindex(lmo.lmos)]
    v = BlockVector(blocks, direction.block_sizes, direction.tot_size)
    return v
end

"""
    compute_extreme_point(lmo::ProductLMO, direction::Tuple; kwargs...)

Extreme point computation on Cartesian product, with a direction `(d1, d2, ...)` given as a tuple of directions.
All keyword arguments are passed to all LMOs.
"""
function compute_extreme_point(lmo::ProductLMO, direction::Tuple; kwargs...)
    return compute_extreme_point.(lmo.lmos, direction; kwargs...)
end

"""
    compute_extreme_point(lmo::ProductLMO, direction::AbstractArray; direction_indices, storage=similar(direction))

Extreme point computation, with a direction array and `direction_indices` provided such that:
`direction[direction_indices[i]]` is passed to the i-th LMO.
If no `direction_indices` are provided, the direction array is sliced along the last dimension and such that:
`direction[:, ... ,:, i]` is passed to the i-th LMO.
The result is stored in the optional `storage` container.

All keyword arguments are passed to all LMOs.
"""
function compute_extreme_point(
    lmo::ProductLMO{N},
    direction::AbstractArray;
    storage=similar(direction),
    direction_indices=nothing,
    kwargs...,
) where {N}
    if direction_indices !== nothing
        for idx in 1:N
            storage[direction_indices[idx]] .=
                compute_extreme_point(lmo.lmos[idx], direction[direction_indices[idx]]; kwargs...)
        end
    else
        ndim = ndims(direction)
        direction_array = [direction[[idx < ndim ? Colon() : i for idx in 1:ndim]...] for i in 1:N]
        storage = cat(compute_extreme_point.(lmo.lmos, direction_array)..., dims=ndim)
    end
    return storage
end

"""
MathOptInterface LMO but returns a vertex respecting the block structure
"""
function FrankWolfe.compute_extreme_point(lmo::FrankWolfe.MathOptLMO, direction::BlockVector)
    xs = MOI.get(lmo.o, MOI.ListOfVariableIndices())
    terms = [MOI.ScalarAffineTerm(direction[idx], xs[idx]) for idx in eachindex(xs)]
    vec_v = FrankWolfe.compute_extreme_point(lmo::FrankWolfe.MathOptLMO, terms)
    v = similar(direction)
    copyto!(v, vec_v)
    return v
end

function FrankWolfe.muladd_memory_mode(mem::FrankWolfe.InplaceEmphasis, storage::BlockVector, x::BlockVector, gamma::Real, d::BlockVector)
    @inbounds for i in eachindex(x.blocks)
        FrankWolfe.muladd_memory_mode(mem, storage.blocks[i], x.blocks[i], gamma, d.blocks[i])
    end
    return storage
end

function FrankWolfe.muladd_memory_mode(mem::FrankWolfe.InplaceEmphasis, x::BlockVector, gamma::Real, d::BlockVector)
    @inbounds for i in eachindex(x.blocks)
        FrankWolfe.muladd_memory_mode(mem, x.blocks[i], gamma, d.blocks[i])
    end
    return x
end

function FrankWolfe.muladd_memory_mode(mem::FrankWolfe.InplaceEmphasis, d::BlockVector, x::BlockVector, v::BlockVector)
    @inbounds for i in eachindex(d.blocks)
        FrankWolfe.muladd_memory_mode(mem, d.blocks[i], x.blocks[i], v.blocks[i])
    end
    return d
end

function FrankWolfe.compute_active_set_iterate!(active_set::FrankWolfe.ActiveSet{<:BlockVector})
    @inbounds for i in eachindex(active_set.x.blocks)
        @. active_set.x.blocks[i] .= 0
    end
    for (λi, ai) in active_set
        for i in eachindex(active_set.x.blocks)
            FrankWolfe.muladd_memory_mode(FrankWolfe.InplaceEmphasis(), active_set.x.blocks[i], -λi, ai.blocks[i])
        end
    end
    return active_set.x
end
