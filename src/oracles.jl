
"""
Supertype for linear minimization oracles.

All LMOs must implement `compute_extreme_point(lmo::LMO, direction)`
and return a vector `v` of the appropriate type.
"""
abstract type LinearMinimizationOracle end

"""
    compute_extreme_point(lmo::LinearMinimizationOracle, direction; kwargs...)

Computes the point `argmin_{v ∈ C} v ⋅ direction`
with `C` the set represented by the LMO.
All LMOs should accept keyword arguments that they can ignore.
"""
function compute_extreme_point end

"""
    CachedLinearMinimizationOracle{LMO}

Oracle wrapping another one of type lmo.
Subtypes of `CachedLinearMinimizationOracle` contain a cache of
previous solutions.

By convention, the inner oracle is named `inner`.
Cached optimizers are expected to implement `Base.empty!` and `Base.length`.
"""
abstract type CachedLinearMinimizationOracle{LMO<:LinearMinimizationOracle} <:
              LinearMinimizationOracle end

# by default do nothing and return the LMO itself
Base.empty!(lmo::CachedLinearMinimizationOracle) = lmo
Base.length(::CachedLinearMinimizationOracle) = 0

"""
    SingleLastCachedLMO{LMO, VT}

Caches only the last result from an LMO and stores it in `last_vertex`.
Vertices of `LMO` have to be of type `VT` if provided.
"""
mutable struct SingleLastCachedLMO{LMO,A} <: CachedLinearMinimizationOracle{LMO}
    last_vertex::Union{Nothing,A}
    inner::LMO
end

# initializes with no cache by default
SingleLastCachedLMO(lmo::LMO) where {LMO<:LinearMinimizationOracle} =
    SingleLastCachedLMO{LMO,AbstractVector}(nothing, lmo)

function compute_extreme_point(
    lmo::SingleLastCachedLMO,
    direction;
    threshold=-Inf,
    store_cache=true,
    kwargs...,
)
    if lmo.last_vertex !== nothing && isfinite(threshold)
        v = lmo.last_vertex
        if fast_dot(v, direction) ≤ threshold # cache is a sufficiently-decreasing direction
            return v
        end
    end
    v = compute_extreme_point(lmo.inner, direction; kwargs...)
    if store_cache
        lmo.last_vertex = v
    end
    return v
end

function Base.empty!(lmo::SingleLastCachedLMO)
    lmo.last_vertex = nothing
    return lmo
end

Base.length(lmo::SingleLastCachedLMO) = Int(lmo.last_vertex !== nothing)

"""
    MultiCacheLMO{N, LMO, VT}

Cache for a LMO storing up to `N` vertices in the cache, removed in FIFO style.
`oldest_idx` keeps track of the oldest index in the tuple, i.e. to replace next.
`VT`, if provided, must be the type of vertices returned by `LMO`
"""
mutable struct MultiCacheLMO{N,LMO<:LinearMinimizationOracle,A} <:
               CachedLinearMinimizationOracle{LMO}
    vertices::NTuple{N,Union{A,Nothing}}
    inner::LMO
    oldest_idx::Int
end

function MultiCacheLMO{N,LMO,A}(lmo::LMO) where {N,LMO<:LinearMinimizationOracle,A}
    return MultiCacheLMO{N,LMO,A}(ntuple(_ -> nothing, Val{N}()), lmo, 1)
end

function MultiCacheLMO{N}(lmo::LMO) where {N,LMO<:LinearMinimizationOracle}
    return MultiCacheLMO{N,LMO,AbstractVector}(ntuple(_ -> nothing, Val{N}()), lmo, 1)
end

# arbitrary default to 10 points
function MultiCacheLMO(lmo::LMO) where {LMO<:LinearMinimizationOracle}
    return MultiCacheLMO{10}(lmo)
end

# type-unstable
function MultiCacheLMO(n::Integer, lmo::LMO) where {LMO<:LinearMinimizationOracle}
    return MultiCacheLMO{n}(lmo)
end

function Base.empty!(lmo::MultiCacheLMO{N}) where {N}
    lmo.vertices = ntuple(_ -> nothing, Val{N}())
    lmo.oldest_idx = 1
    return lmo
end

Base.length(lmo::MultiCacheLMO) = count(!isnothing, lmo.vertices)

"""
Compute the extreme point with a multi-vertex cache.
`store_cache` indicates whether the newly-computed point should be stored in cache.
`greedy` determines if we should return the first point with dot-product
below `threshold` or look for the best one.
"""
function compute_extreme_point(
    lmo::MultiCacheLMO{N},
    direction;
    threshold=-Inf,
    store_cache=true,
    greedy=false,
    kwargs...,
) where {N}
    if isfinite(threshold)
        best_idx = -1
        best_val = Inf
        best_v = nothing
        # create an iteration order to visit most recent vertices first
        iter_order = if lmo.oldest_idx > 1
            Iterators.flatten((lmo.oldest_idx-1:-1:1, N:-1:lmo.oldest_idx))
        else
            N:-1:1
        end
        for idx in iter_order
            if lmo.vertices[idx] !== nothing
                v = lmo.vertices[idx]
                new_val = fast_dot(v, direction)
                if new_val ≤ threshold # cache is a sufficiently-decreasing direction
                    # if greedy, stop and return point
                    if greedy
                        # println("greedy cache sol")
                        return v
                    end
                    # otherwise, keep the index only if better than incumbent
                    if new_val < best_val
                        best_idx = idx
                        best_val = new_val
                        best_v = v
                    end
                end
            end
        end
        if best_idx > 0 # && fast_dot(best_v, direction) ≤ threshold
            # println("cache sol")
            return best_v
        end
    end
    # no interesting point found, computing new
    # println("LP sol")
    v = compute_extreme_point(lmo.inner, direction; kwargs...)
    if store_cache
        tup = Base.setindex(lmo.vertices, v, lmo.oldest_idx)
        lmo.vertices = tup
        # if oldest_idx was last, we get back to 1, otherwise we increment oldest index
        lmo.oldest_idx = lmo.oldest_idx < N ? lmo.oldest_idx + 1 : 1
    end
    return v
end


"""
    VectorCacheLMO{N, LMO, VT}

Cache for a LMO storing an unbounded number of vertices of type `VT` in the cache.
`VT`, if provided, must be the type of vertices returned by `LMO`
"""
mutable struct VectorCacheLMO{LMO<:LinearMinimizationOracle,VT} <:
               CachedLinearMinimizationOracle{LMO}
    vertices::Vector{VT}
    inner::LMO
end

function VectorCacheLMO{LMO,VT}(lmo::LMO) where {VT,LMO<:LinearMinimizationOracle}
    return VectorCacheLMO{LMO,VT}(VT[], lmo)
end

function VectorCacheLMO(lmo::LMO) where {LMO<:LinearMinimizationOracle}
    return VectorCacheLMO{LMO,Vector{Float64}}(AbstractVector[], lmo)
end

function Base.empty!(lmo::VectorCacheLMO)
    empty!(lmo.vertices)
    return lmo
end

Base.length(lmo::VectorCacheLMO) = length(lmo.vertices)

function compute_extreme_point(
    lmo::VectorCacheLMO,
    direction;
    threshold=-Inf,
    store_cache=true,
    greedy=false,
    kwargs...,
)
    if isempty(lmo.vertices)
        v = compute_extreme_point(lmo.inner, direction)
        if store_cache
            push!(lmo.vertices, v)
        end
        return v
    end
    best_idx = -1
    best_val = Inf
    best_v = nothing
    for idx in reverse(eachindex(lmo.vertices))
        @inbounds v = lmo.vertices[idx]
        new_val = fast_dot(v, direction)
        if new_val ≤ threshold
            # stop, store and return
            if greedy
                return v
            end
            # otherwise, compare to incumbent
            if new_val < best_val
                best_v = v
                best_val = new_val
                best_idx = idx
            end
        end
    end
    v = best_v
    if best_idx < 0
        v = compute_extreme_point(lmo.inner, direction)
        if store_cache
            # note: we do not check for duplicates. hence you might end up with more vertices,
            # in fact up to number of dual steps many, that might be already in the cache
            # in order to reach this point, if v was already in the cache is must not meet the threshold (otherwise we would have returned it)
            # and it is the best possible, hence we will perform a dual step on the outside.
            #
            # note: another possibility could be to test against that in the if statement but then you might end you recalculating the same vertex a few times.
            # as such this might be a better tradeoff, i.e., to not check the set for duplicates and potentially accept #dualSteps many duplicates.
            push!(lmo.vertices, v)
        end
    end
    return v
end

"""
    ProductLMO(lmos...)

Linear minimization oracle over the Cartesian product of multiple LMOs.
"""
struct ProductLMO{N,TL<:NTuple{N,LinearMinimizationOracle}} <: LinearMinimizationOracle
    lmos::TL
end

function ProductLMO{N}(lmos::TL) where {N,TL<:NTuple{N,LinearMinimizationOracle}}
    return ProductLMO{N,TL}(lmos)
end

function ProductLMO(lmos::Vararg{LinearMinimizationOracle,N}) where {N}
    return ProductLMO{N}(lmos)
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
The result is stored in the optional `storage` container.

All keyword arguments are passed to all LMOs.
"""
function compute_extreme_point(
    lmo::ProductLMO{N},
    direction::AbstractArray;
    storage=similar(direction),
    direction_indices,
    kwargs...,
) where {N}
    for idx in 1:N
        storage[direction_indices[idx]] .=
            compute_extreme_point(lmo.lmos[idx], direction[direction_indices[idx]]; kwargs...)
    end
    return storage
end

"""
    ChasingGradientLMO{LMO,T}

Oracle wrapping another one of type LMO to boost FW with gradient pursuit.
Pursuit rounds end once the alignment improvement with the direction of the gradient
is lower than `improv_tol` or `max_rounds_number` is reached.
See the [paper](https://arxiv.org/abs/2003.06369)

All keyword arguments are passed to the inner LMO.
"""
mutable struct ChasingGradientLMO{LMO<:LinearMinimizationOracle,T,G,G2} <: LinearMinimizationOracle
    inner::LMO
    max_rounds_number::Int
    improv_tol::T
    d::G
    u::G
    residual::G2
    m_residual::G2
end

function ChasingGradientLMO(lmo, max_rounds_number, improv_tol, d, residual)
    return ChasingGradientLMO(
        lmo,
        max_rounds_number,
        improv_tol,
        d,
        copy(d),
        residual,
        copy(residual),
    )
end

function _zero!(d)
    @. d = 0
end

function _zero!(d::SparseArrays.AbstractSparseArray)
    @. d = 0
    return SparseArrays.dropzeros!(d)
end

function _inplace_plus(d, v)
    @. d += v
end

function _inplace_plus(d, v::ScaledHotVector)
    return d[v.val_idx] += v.active_val
end

function compute_extreme_point(lmo::ChasingGradientLMO, direction; x, kwargs...)
    _zero!(lmo.d)
    v_ret = 0 * lmo.d
    Λ = zero(eltype(direction))
    norm_direction = norm(direction)
    if norm_direction <= eps(eltype(direction))
        v_ret .+= compute_extreme_point(lmo.inner, direction, x=x, kwargs...)
        return v_ret
    end
    use_v = true
    function align(d)
        if norm(d) <= eps(eltype(d))
            return -one(eltype(d))
        else
            return -fast_dot(direction, d) / (norm_direction * norm(d))
        end
    end
    @. lmo.residual = -direction
    @. lmo.m_residual = direction
    for round in 1:lmo.max_rounds_number
        if norm(lmo.residual) <= eps(eltype(lmo.residual))
            #@show round
            break
        end
        v = compute_extreme_point(lmo.inner, lmo.m_residual; kwargs...)
        #@. lmo.u = v - x
        @. lmo.u = -x
        _inplace_plus(lmo.u, v)
        d_norm = norm(lmo.d)
        if d_norm > 0 && fast_dot(lmo.residual, lmo.d / d_norm) < -fast_dot(lmo.residual, lmo.u)
            @. lmo.u = -lmo.d / d_norm
            use_v = false
        end
        λ = fast_dot(lmo.residual, lmo.u) / (norm(lmo.u)^2)
        d_updated = lmo.u
        @. d_updated = lmo.d + λ * lmo.u #2 allocs
        if align(d_updated) - align(lmo.d) >= lmo.improv_tol
            @. lmo.d = d_updated
            # @. lmo.residual = -direction - lmo.d
            @. lmo.residual -= λ * lmo.u
            @. lmo.m_residual = -lmo.residual
            if use_v
                Λ += λ
            else
                Λ = Λ * (1 - λ / d_norm)
            end
            use_v = true
        else
            #@show round
            break
        end
    end
    if Λ <= eps(eltype(Λ))
        v_ret .+= compute_extreme_point(lmo.inner, direction, x=x, kwargs...)
        return v_ret
    end
    @. v_ret += x
    @. v_ret += lmo.d / Λ #2 allocations
    return v_ret
end
