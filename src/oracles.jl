
"""
Supertype for linear minimization oracles.

All LMOs must implement `compute_extreme_point(lmo::LMO, direction)`
and return a vector `v` of the appropriate type.
"""
abstract type LinearMinimizationOracle
end

"""
    compute_extreme_point(lmo::LinearMinimizationOracle, direction; kwargs...)

Computes the point `argmin_{v ∈ C} v ⋅ direction`
with `C` the set represented by the LMO.
"""
function compute_extreme_point end

"""
    CachedLinearMinimizationOracle{LMO}

Oracle wrapping another one of type lmo.
Subtypes of `CachedLinearMinimizationOracle` contain a cache of
previous solutions.

By convention, the inner oracle is named `inner`.
"""
abstract type CachedLinearMinimizationOracle{LMO <: LinearMinimizationOracle} <: LinearMinimizationOracle
end

"""
    SingleLastCachedLMO{LMO, VT}

Caches only the last result from an LMO and stores it in `last_vertex`.
Vertices of `LMO` have to be of type `VT` if provided.
"""
mutable struct SingleLastCachedLMO{LMO, VT <: AbstractVector} <: CachedLinearMinimizationOracle{LMO}
    last_vertex::Union{Nothing, VT}
    inner::LMO
end

# initializes with no cache by default
SingleLastCachedLMO(lmo::LMO) where {LMO <: LinearMinimizationOracle} = SingleLastCachedLMO{LMO, AbstractVector}(nothing, lmo)

function compute_extreme_point(lmo::SingleLastCachedLMO, direction; threshold=-Inf, store_cache=true, kwargs...)
    if lmo.last_vertex !== nothing && isfinite(threshold)
        v = lmo.last_vertex
        if dot(v, direction) ≤ threshold # cache is a sufficiently-decreasing direction
            return v
        end
    end
    v = compute_extreme_point(lmo.inner, direction, kwargs...)
    if store_cache
        lmo.last_vertex = v
    end
    return v
end

"""
    MultiCacheLMO{N, LMO, VT}

Cache for a LMO storing up to `N` vertices in the cache, removed in FIFO style.
`oldest_idx` keeps track of the oldest index in the tuple, i.e. to replace next.
`VT`, if provided, must be the type of vertices returned by `LMO`
"""
mutable struct MultiCacheLMO{N, LMO, VT <: AbstractVector} <: CachedLinearMinimizationOracle{LMO}
    vertices::NTuple{N, Union{VT, Nothing}}
    inner::LMO
    oldest_idx::Int
end

function MultiCacheLMO{N}(lmo::LMO) where {N, LMO <: LinearMinimizationOracle}
    return MultiCacheLMO{N, LMO, AbstractVector}(
        ntuple(_->nothing, Val{N}()),
        lmo,
        1,
    )
end

# arbitrary default to 10 points
function MultiCacheLMO(lmo::LMO) where {LMO <: LinearMinimizationOracle}
    return MultiCacheLMO{10}(lmo)
end

# type-unstable
function MultiCacheLMO(n::Integer, lmo::LMO) where {LMO <: LinearMinimizationOracle}
    return MultiCacheLMO{n}(lmo)
end

"""
Compute the extreme point with a multi-vertex cache.
`store_cache` indicates whether the newly-computed point should be stored in cache.
`greedy` determines if we should return the first point with dot-product
below `threshold` or look for the best one.
"""
function compute_extreme_point(lmo::MultiCacheLMO{N}, direction; threshold=-Inf, store_cache=true, greedy=false, kwargs...) where {N}
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
                new_val = dot(v, direction)
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
        if best_idx > 0 # && dot(best_v, direction) ≤ threshold 
            # println("cache sol")
            return best_v
        end
    end
    # no interesting point found, computing new
    # println("LP sol")
    v = compute_extreme_point(lmo.inner, direction, kwargs...)
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
mutable struct VectorCacheLMO{LMO, VT <: AbstractVector} <: CachedLinearMinimizationOracle{LMO}
    vertices::Vector{VT}
    inner::LMO
end

function VectorCacheLMO{VT}(lmo::LMO) where {VT, LMO <: LinearMinimizationOracle}
    return VectorCacheLMO{LMO, VT}(VT[], lmo)
end

function VectorCacheLMO(lmo::LMO) where {LMO <: LinearMinimizationOracle}
    return VectorCacheLMO{LMO, AbstractVector}(AbstractVector[], lmo)
end

function compute_extreme_point(lmo::VectorCacheLMO, direction; threshold=-Inf, store_cache=true, greedy=false, kwargs...)
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
        new_val = dot(v, direction)
        if new_val ≤ threshold
            # stop, store and return
            if greedy
                if store_cache
                    push!(lmo.vertices, v)
                end
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
    end
    if store_cache
        push!(lmo.vertices, v)
    end
    return v
end
