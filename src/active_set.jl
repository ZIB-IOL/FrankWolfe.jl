
"""
    AbstractActiveSet{AT, R, IT}

Abstract type for an active set of atoms of type `AT` with weights of type `R` and iterate of type `IT`.
An active set is typically expected to have a field `weights`, a field `atoms`, and a field `x`.
Otherwise, all active set methods from `src/active_set.jl` can be overwritten.
"""
abstract type AbstractActiveSet{AT,R<:Real,IT} <: AbstractVector{Tuple{R,AT}} end

"""
    ActiveSet{AT, R, IT}

Represents an active set of extreme vertices collected in a FW algorithm,
along with their coefficients `(λ_i, a_i)`.
`R` is the type of the `λ_i`, `AT` is the type of the atoms `a_i`.
The iterate `x = ∑λ_i a_i` is stored in x with type `IT`.
"""
struct ActiveSet{AT,R<:Real,IT} <: AbstractActiveSet{AT,R,IT}
    weights::Vector{R}
    atoms::Vector{AT}
    x::IT
end

ActiveSet{AT,R}() where {AT,R} = ActiveSet{AT,R,Vector{float(eltype(AT))}}([], [])

ActiveSet{AT}() where {AT} = ActiveSet{AT,Float64,Vector{float(eltype(AT))}}()

function ActiveSet(tuple_values::AbstractVector{Tuple{R,AT}}) where {AT,R}
    return ActiveSet{AT,R}(tuple_values)
end

function ActiveSet{AT,R}(tuple_values::AbstractVector{<:Tuple{<:Number,<:Any}}) where {AT,R}
    n = length(tuple_values)
    weights = Vector{R}(undef, n)
    atoms = Vector{AT}(undef, n)
    @inbounds for idx in 1:n
        weights[idx] = tuple_values[idx][1]
        atoms[idx] = tuple_values[idx][2]
    end
    x = similar(atoms[1], float(eltype(atoms[1])))
    as = ActiveSet{AT,R,typeof(x)}(weights, atoms, x)
    compute_active_set_iterate!(as)
    return as
end

Base.getindex(as::AbstractActiveSet, i) = (as.weights[i], as.atoms[i])
Base.size(as::AbstractActiveSet) = size(as.weights)

# these three functions do not update the active set iterate

function Base.push!(as::AbstractActiveSet, (λ, a))
    push!(as.weights, λ)
    push!(as.atoms, a)
    return as
end

function Base.deleteat!(as::AbstractActiveSet, idx)
    # WARNING assumes that idx is sorted
    for (i, j) in enumerate(idx)
        deleteat!(as, j - i + 1)
    end
    return as
end

function Base.deleteat!(as::AbstractActiveSet, idx::Int)
    deleteat!(as.atoms, idx)
    deleteat!(as.weights, idx)
    return as
end

function Base.empty!(as::AbstractActiveSet)
    empty!(as.atoms)
    empty!(as.weights)
    as.x .= 0
    return as
end

function Base.isempty(as::AbstractActiveSet)
    return isempty(as.atoms)
end

"""
Copies an active set, the weight and atom vectors and the iterate.
Individual atoms are not copied.
"""
function Base.copy(as::AbstractActiveSet{AT,R,IT}) where {AT,R,IT}
    return ActiveSet{AT,R,IT}(copy(as.weights), copy(as.atoms), copy(as.x))
end

"""
    active_set_update!(active_set::AbstractActiveSet, lambda, atom)

Adds the atom to the active set with weight lambda or adds lambda to existing atom.
"""
function active_set_update!(
    active_set::AbstractActiveSet{AT,R},
    lambda,
    atom,
    renorm=true,
    idx=nothing;
    weight_purge_threshold=weight_purge_threshold_default(R),
    add_dropped_vertices=false,
    vertex_storage=nothing,
) where {AT,R}
    # rescale active set
    active_set_mul_weights!(active_set, 1 - lambda)
    # add value for new atom
    if idx === nothing
        idx = find_atom(active_set, atom)
    end
    if idx > 0
        active_set_add_weight!(active_set, lambda, idx)
    else
        push!(active_set, (lambda, atom))
    end
    if renorm
        add_dropped_vertices =
            add_dropped_vertices ? vertex_storage !== nothing : add_dropped_vertices
        active_set_cleanup!(
            active_set;
            weight_purge_threshold=weight_purge_threshold,
            update=false,
            add_dropped_vertices=add_dropped_vertices,
            vertex_storage=vertex_storage,
        )
        active_set_renormalize!(active_set)
    end
    active_set_update_scale!(active_set.x, lambda, atom)
    return active_set
end

"""
    active_set_update_scale!(x, lambda, atom)

Operates `x ← (1-λ) x + λ a`.
"""
function active_set_update_scale!(x::IT, lambda, atom) where {IT}
    @. x = x * (1 - lambda) + lambda * atom
    return x
end

function active_set_update_scale!(x::IT, lambda, atom::SparseArrays.SparseVector) where {IT}
    @. x *= (1 - lambda)
    nzvals = SparseArrays.nonzeros(atom)
    nzinds = SparseArrays.nonzeroinds(atom)
    @inbounds for idx in eachindex(nzvals)
        x[nzinds[idx]] += lambda * nzvals[idx]
    end
    return x
end

"""
    active_set_update_pairwise!(active_set, gamma, gamma_max, v_local_loc, a_loc, v_local, a, add_dropped_vertices, extra_vertex_storage)

Updates the active set for a pairwise step with step size gamma.
"""
function active_set_update_pairwise!(
    active_set::AbstractActiveSet{AT,R},
    gamma::Real,
    gamma_max::Real,
    v_local_loc::Integer,
    a_loc::Integer,
    v_local::AT,
    a::AT,
    add_dropped_vertices::Bool,
    extra_vertex_storage=nothing,
) where {AT,R}
    # reached maximum of lambda -> dropping away vertex
    if gamma ≈ gamma_max
        active_set_add_weight!(active_set, gamma, v_local_loc)
        deleteat!(active_set, a_loc)
        if add_dropped_vertices
            push!(extra_vertex_storage, a)
        end
    else # transfer weight from away to local FW
        active_set_add_weight!(active_set, -gamma, a_loc)
        active_set_add_weight!(active_set, gamma, v_local_loc)
        @assert active_set_validate(active_set)
    end
    active_set_update_iterate_pairwise!(active_set.x, gamma, v_local, a)
    return active_set
end

"""
    active_set_mul_weights!(active_set, lambda)

Multiplies all weights in `active_set` by `lambda`.
"""
function active_set_mul_weights!(active_set::AbstractActiveSet, lambda::Real)
    @inbounds active_set.weights .*= lambda
end

"""
    active_set_add_weight!(active_set, lambda, i)

Adds `lambda` to the weight of the `i`th atom in `active_set`.
"""
function active_set_add_weight!(active_set::AbstractActiveSet, lambda::Real, i::Integer)
    return active_set.weights[i] += lambda
end

"""
    active_set_update_iterate_pairwise!(active_set, x, lambda, fw_atom, away_atom)

Operates `x ← x + λ a_fw - λ a_aw`.
"""
function active_set_update_iterate_pairwise!(
    x::IT,
    lambda::Real,
    fw_atom::A,
    away_atom::A,
) where {IT,A}
    @. x += lambda * fw_atom - lambda * away_atom
    return x
end

function active_set_validate(active_set::AbstractActiveSet)
    return sum(active_set.weights) ≈ 1.0 && all(≥(0), active_set.weights)
end

function active_set_renormalize!(active_set::AbstractActiveSet)
    renorm = sum(active_set.weights)
    active_set_mul_weights!(active_set, inv(renorm))
    return active_set
end

function weight_from_atom(active_set::AbstractActiveSet, atom)
    idx = find_atom(active_set, atom)
    if idx > 0
        return active_set.weights[idx]
    else
        return nothing
    end
end

"""
    get_active_set_iterate(active_set)

Return the current iterate corresponding. Does not recompute it.
"""
function get_active_set_iterate(active_set)
    return active_set.x
end

"""
    compute_active_set_iterate!(active_set::AbstractActiveSet) -> x

Recomputes from scratch the iterate `x` from the current weights and vertices of the active set.
Returns the iterate `x`.
"""
function compute_active_set_iterate!(active_set)
    active_set.x .= 0
    for (λi, ai) in active_set
        @. active_set.x += λi * ai
    end
    return active_set.x
end

# specialized version for sparse vector
function compute_active_set_iterate!(active_set::AbstractActiveSet{<:SparseArrays.SparseVector})
    active_set.x .= 0
    for (λi, ai) in active_set
        nzvals = SparseArrays.nonzeros(ai)
        nzinds = SparseArrays.nonzeroinds(ai)
        @inbounds for idx in eachindex(nzvals)
            active_set.x[nzinds[idx]] += λi * nzvals[idx]
        end
    end
    return active_set.x
end

function compute_active_set_iterate!(
    active_set::FrankWolfe.ActiveSet{<:SparseArrays.AbstractSparseMatrix},
)
    active_set.x .= 0
    for (λi, ai) in active_set
        (I, J, V) = SparseArrays.findnz(ai)
        @inbounds for idx in eachindex(I)
            active_set.x[I[idx], J[idx]] += λi * V[idx]
        end
    end
    return active_set.x
end

function active_set_cleanup!(
    active_set::AbstractActiveSet{AT,R};
    weight_purge_threshold=weight_purge_threshold_default(R),
    update=true,
    add_dropped_vertices=false,
    vertex_storage=nothing,
) where {AT,R}
    if add_dropped_vertices && vertex_storage !== nothing
        for (weight, v) in zip(active_set.weights, active_set.atoms)
            if weight ≤ weight_purge_threshold
                push!(vertex_storage, v)
            end
        end
    end
    # one cannot use a generator as deleteat! modifies active_set in place
    deleteat!(
        active_set,
        [idx for idx in eachindex(active_set) if active_set.weights[idx] ≤ weight_purge_threshold],
    )
    if update
        compute_active_set_iterate!(active_set)
    end
    return nothing
end

function find_atom(active_set::AbstractActiveSet, atom)
    @inbounds for idx in eachindex(active_set)
        if _unsafe_equal(active_set.atoms[idx], atom)
            return idx
        end
    end
    return -1
end

"""
    active_set_argmin(active_set::AbstractActiveSet, direction)

Computes the linear minimizer in the direction on the active set.
Returns `(λ_i, a_i, i)`
"""
function active_set_argmin(active_set::AbstractActiveSet, direction)
    valm = typemax(eltype(direction))
    idxm = -1
    @inbounds for i in eachindex(active_set)
        val = dot(active_set.atoms[i], direction)
        if val < valm
            valm = val
            idxm = i
        end
    end
    if idxm == -1
        error(
            "Infinite minimum $valm in the active set. Does the gradient contain invalid (NaN / Inf) entries?",
        )
    end
    return (active_set[idxm]..., idxm)
end

"""
    active_set_argminmax(active_set::AbstractActiveSet, direction)

Computes the linear minimizer in the direction on the active set.
Returns `(λ_min, a_min, i_min, val_min, λ_max, a_max, i_max, val_max, val_max-val_min ≥ Φ)`
"""
function active_set_argminmax(active_set::AbstractActiveSet, direction; Φ=0.5)
    valm = typemax(eltype(direction))
    valM = typemin(eltype(direction))
    idxm = -1
    idxM = -1
    @inbounds for i in eachindex(active_set)
        val = dot(active_set.atoms[i], direction)
        if val < valm
            valm = val
            idxm = i
        end
        if valM < val
            valM = val
            idxM = i
        end
    end
    if idxm == -1 || idxM == -1
        error(
            "Infinite minimum $valm or maximum $valM in the active set. Does the gradient contain invalid (NaN / Inf) entries?",
        )
    end
    return (active_set[idxm]..., idxm, valm, active_set[idxM]..., idxM, valM, valM - valm ≥ Φ)
end

"""
    active_set_initialize!(as, v)

Resets the active set structure to a single vertex `v` with unit weight.
"""
function active_set_initialize!(as::AbstractActiveSet{AT,R}, v) where {AT,R}
    empty!(as)
    push!(as, (one(R), v))
    compute_active_set_iterate!(as)
    return as
end

function compute_active_set_iterate!(
    active_set::AbstractActiveSet{<:ScaledHotVector,<:Real,<:AbstractVector},
)
    active_set.x .= 0
    @inbounds for (λi, ai) in active_set
        active_set.x[ai.val_idx] += λi * ai.active_val
    end
    return active_set.x
end

function update_weights!(as::AbstractActiveSet, new_weights)
    return as.weights .= new_weights
end

"""
Vertex storage to store dropped vertices or find a suitable direction in lazy settings.
The algorithm will look for at most `return_kth` suitable atoms before returning the best.
See [Extra-lazification with a vertex storage](@ref) for usage.

A vertex storage can be any type that implements two operations:
1. `Base.push!(storage, atom)` to add an atom to the storage.
Note that it is the storage type responsibility to ensure uniqueness of the atoms present.
2. `storage_find_argmin_vertex(storage, direction, lazy_threshold) -> (found, vertex)`
returning whether a vertex with sufficient progress was found and the vertex.
It is up to the storage to remove vertices (or not) when they have been picked up.
"""
struct DeletedVertexStorage{AT}
    storage::Vector{AT}
    return_kth::Int
end

DeletedVertexStorage(storage::Vector) = DeletedVertexStorage(storage, 1)
DeletedVertexStorage{AT}() where {AT} = DeletedVertexStorage(AT[])

function Base.push!(vertex_storage::DeletedVertexStorage{AT}, atom::AT) where {AT}
    # do not push duplicates
    if !any(v -> _unsafe_equal(atom, v), vertex_storage.storage)
        push!(vertex_storage.storage, atom)
    end
    return vertex_storage
end

Base.length(storage::DeletedVertexStorage) = length(storage.storage)

"""
Computes the linear minimizer in the direction on the precomputed_set.
Precomputed_set stores the vertices computed as extreme points v in each iteration.
"""
function pre_computed_set_argminmax(lmo, pre_computed_set, direction, x; strong_lazification=false)
    val = convert(eltype(direction), Inf)
    valM = convert(eltype(direction), -Inf)
    idx = -1
    idxM = -1
    for i in eachindex(pre_computed_set)
        temp_val = dot(pre_computed_set[i], direction)
        if temp_val < val
            val = temp_val
            idx = i
        end
        if strong_lazification
            if is_inface_feasible(lmo, pre_computed_set[i], x) && temp_val > valM
                valM = temp_val
                idxM = i
            end
        end
    end
    if idx == -1
        error(
            "Infinite minimum $val in the precomputed set. Does the gradient contain invalid (NaN / Inf) entries?",
        )
    end
    v_local = pre_computed_set[idx]
    a_local = idxM != -1 ? pre_computed_set[idxM] : nothing
    return (v_local, idx, val, a_local, idxM, valM)
end

"""
Give the vertex `v` in the storage that minimizes `s = direction ⋅ v` and whether `s` achieves
`s ≤ lazy_threshold`.
"""
function storage_find_argmin_vertex(vertex_storage::DeletedVertexStorage, direction, lazy_threshold)
    if isempty(vertex_storage.storage)
        return (false, nothing)
    end
    best_idx = 1
    best_val = lazy_threshold
    found_good = false
    counter = 0
    for (idx, atom) in enumerate(vertex_storage.storage)
        s = dot(direction, atom)
        if s < best_val
            counter += 1
            best_val = s
            found_good = true
            best_idx = idx
            if counter ≥ vertex_storage.return_kth
                return (found_good, vertex_storage.storage[best_idx])
            end
        end
    end
    return (found_good, vertex_storage.storage[best_idx])
end

"""
    _unsafe_equal(a, b)

Like `isequal` on arrays but without the checks. Assumes a and b have the same axes.
"""
function _unsafe_equal(a::Array, b::Array)
    if a === b
        return true
    end
    @inbounds for idx in eachindex(a)
        if a[idx] != b[idx]
            return false
        end
    end
    return true
end

_unsafe_equal(a, b) = isequal(a, b)

function _unsafe_equal(a::SparseArrays.AbstractSparseArray, b::SparseArrays.AbstractSparseArray)
    return a == b
end

function weight_purge_threshold_default(::Type{T}) where {T<:AbstractFloat}
    return sqrt(eps(T) * Base.rtoldefault(T)) # around 1e-12 for Float64
end
weight_purge_threshold_default(::Type{T}) where {T<:Number} = Base.rtoldefault(T)
