
"""
    ActiveSet{AT, R, IT}

Represents an active set of extreme vertices collected in a FW algorithm,
along with their coefficients `(λ_i, a_i)`.
`R` is the type of the `λ_i`, `AT` is the type of the atoms `a_i`.
The iterate `x = ∑λ_i a_i` is stored in x with type `IT`.
"""
struct ActiveSet{AT,R,IT} <: AbstractVector{Tuple{R,AT}}
    weights::Vector{R}
    atoms::Vector{AT}
    x::IT
end

ActiveSet{AT,R}() where {AT,R} = ActiveSet{AT,R,Vector{float(eltype(AT))}}([], [])

ActiveSet{AT}() where {AT} = ActiveSet{AT,Float64,Vector{float(eltype(AT))}}()

function ActiveSet(tuple_values::AbstractVector{Tuple{R,AT}}) where {AT,R}
    n = length(tuple_values)
    weights = Vector{R}(undef, n)
    atoms = Vector{AT}(undef, n)
    @inbounds for idx in 1:n
        weights[idx] = tuple_values[idx][1]
        atoms[idx] = tuple_values[idx][2]
    end
    x = float.(similar(atoms[1]))
    as = ActiveSet{AT,R,typeof(x)}(weights, atoms, x)
    compute_active_set_iterate!(as)
    return as
end

function ActiveSet{AT,R}(tuple_values::AbstractVector{<:Tuple{<:Number,<:Any}}) where {AT,R}
    n = length(tuple_values)
    weights = Vector{R}(undef, n)
    atoms = Vector{AT}(undef, n)
    x = float.(similar(tuple_values[1][2]))
    x .= 0
    @inbounds for idx in 1:n
        weights[idx] = tuple_values[idx][1]
        atoms[idx] = tuple_values[idx][2]
        x .+= weights[idx] * atoms[idx]
    end
    return ActiveSet{AT,R,typeof(x)}(weights, atoms, x)
end

Base.getindex(as::ActiveSet, i) = (as.weights[i], as.atoms[i])
Base.size(as::ActiveSet) = size(as.weights)

function Base.push!(as::ActiveSet, (λ, a))
    push!(as.weights, λ)
    push!(as.atoms, a)
    return as
end

# these two functions do not update the active set iterate

function Base.deleteat!(as::ActiveSet, idx)
    deleteat!(as.weights, idx)
    deleteat!(as.atoms, idx)
    return as
end

function Base.setindex!(as::ActiveSet, tup::Tuple, idx)
    as.weights[idx] = tup[1]
    as.atoms[idx] = tup[2]
    return tup
end

function Base.empty!(as::ActiveSet)
    empty!(as.atoms)
    empty!(as.weights)
    as.x .= 0
    return as
end

function Base.isempty(as::ActiveSet)
    return isempty(as.atoms)
end

"""
Copies an active set, the weight and atom vectors and the iterate.
Individual atoms are not copied.
"""
function Base.copy(as::ActiveSet{AT, R, IT}) where {AT, R, IT}
    return ActiveSet{AT, R, IT}(
        copy(as.weights),
        copy(as.atoms),
        copy(as.x),
    )
end

"""
    active_set_update!(active_set::ActiveSet, lambda, atom)

Adds the atom to the active set with weight lambda or adds lambda to existing atom.
"""
function active_set_update!(active_set::ActiveSet, lambda, atom, renorm=true, idx=nothing)
    # rescale active set
    active_set.weights .*= (1 - lambda)
    # add value for new atom
    if idx === nothing
        idx = find_atom(active_set, atom)
    end
    updating = false
    if idx > 0
        @inbounds active_set.weights[idx] = active_set.weights[idx] + lambda
        updating = true
    else
        push!(active_set, (lambda, atom))
    end
    if renorm
        active_set_cleanup!(active_set, update=false)
        active_set_renormalize!(active_set)
    end
    @. active_set.x = active_set.x * (1 - lambda) + lambda * atom
    return active_set
end

function active_set_update_iterate_pairwise!(active_set::ActiveSet, lambda, fw_atom, away_atom)
    @. active_set.x += lambda * fw_atom - lambda * away_atom
    return active_set
end

function active_set_validate(active_set::ActiveSet)
    return sum(active_set.weights) ≈ 1.0
end

function active_set_renormalize!(active_set::ActiveSet)
    renorm = sum(active_set.weights)
    active_set.weights ./= renorm
    return active_set
end

function weight_from_atom(active_set::ActiveSet, atom)
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

function compute_active_set_iterate!(active_set)
    active_set.x .= 0
    for (λi, ai) in active_set
        active_set.x .+= λi * ai
    end
    return active_set.x
end

function active_set_cleanup!(active_set; weight_purge_threshold=1e-12, update=true)
    filter!(e -> e[1] > weight_purge_threshold, active_set)
    if update
        compute_active_set_iterate!(active_set)
    end
    return nothing
end

function find_atom(active_set::ActiveSet, atom)
    @inbounds for idx in eachindex(active_set)
        if _unsafe_equal(active_set.atoms[idx], atom)
            return idx
        end
    end
    return -1
end

"""
    active_set_argmin(active_set::ActiveSet, direction)

Computes the linear minimizer in the direction on the active set.
Returns `(λ_i, a_i, i)`
"""
function active_set_argmin(active_set::ActiveSet, direction)
    val = dot(active_set.atoms[1], direction)
    idx = 1
    temp = 0
    for i in 2:length(active_set)
        temp = fast_dot(active_set.atoms[i], direction)
        if temp < val
            val = temp
            idx = i
        end
    end
    # return lambda, vertex, index
    return (active_set[idx]..., idx)
end

"""
    active_set_argminmax(active_set::ActiveSet, direction)

Computes the linear minimizer in the direction on the active set.
Returns `(λ_min, a_min, i_min, val_min, λ_max, a_max, i_max, val_max, val_max-val_min ≥ Φ)`
"""
function active_set_argminmax(active_set::ActiveSet, direction; Φ=0.5)
    val = Inf
    valM = -Inf
    idx = -1
    idxM = -1
    for i in eachindex(active_set)
        temp_val = fast_dot(active_set.atoms[i], direction)
        if temp_val < val
            val = temp_val
            idx = i
        end
        if valM < temp_val
            valM = temp_val
            idxM = i
        end
    end
    return (active_set[idx]..., idx, val, active_set[idxM]..., idxM, valM, valM - val ≥ Φ)
end


"""
    active_set_initialize!(as, v)

Resets the active set structure to a single vertex `v` with unit weight.
"""
function active_set_initialize!(as::ActiveSet{AT,R}, v) where {AT,R}
    empty!(as)
    push!(as, (one(R), v))
    @. as.x = one(R) * v
    return as
end
