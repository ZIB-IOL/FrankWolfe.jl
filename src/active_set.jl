
# TODO: make much nicer with structs etc. also the active_set_return_iterate function needs to 
# be modular for non-vector objects and potentially overloaded

"""
    ActiveSet{AT, R}

Represents an active set of extreme vertices collected in a FW algorithm,
along with their coefficients `(λ_i, a_i)`.
`R` is the type of the `λ_i`, `AT` is the type of the atoms `a_i`.
"""
struct ActiveSet{AT, R} <: AbstractVector{Tuple{R, AT}}
    weights::Vector{R}
    atoms::Vector{AT}
end

ActiveSet{AT, R}() where {AT, R} = ActiveSet{AT, R}([], [])

ActiveSet{AT}() where {AT} = ActiveSet{AT, Float64}()

function ActiveSet(tuple_values::AbstractVector{Tuple{R, AT}}) where {AT, R}
    n = length(tuple_values)
    weights = Vector{R}(undef, n)
    atoms = Vector{AT}(undef, n)
    @inbounds for idx in 1:n
        weights[idx] = tuple_values[idx][1]
        atoms[idx] = tuple_values[idx][2]
    end
    return ActiveSet{AT, R}(weights, atoms)
end

function ActiveSet{AT, R}(tuple_values::AbstractVector{<:NTuple{2}}) where {AT, R}
    n = length(tuple_values)
    weights = Vector{R}(undef, n)
    atoms = Vector{AT}(undef, n)
    @inbounds for idx in 1:n
        weights[idx] = tuple_values[idx][1]
        atoms[idx] = tuple_values[idx][2]
    end
    return ActiveSet{AT, R}(weights, atoms)
end

Base.getindex(as::ActiveSet, i) = (as.weights[i], as.atoms[i])
Base.size(as::ActiveSet) = size(as.weights)

function Base.push!(as::ActiveSet, (λ, a))
    push!(as.weights, λ)
    push!(as.atoms, a)
    as
end

function Base.deleteat!(as::ActiveSet, idx)
    deleteat!(as.weights, idx)
    deleteat!(as.atoms, idx)
    return as
end

function Base.setindex!(as::ActiveSet, tup::Tuple, idx)
    as.weights[idx] = tup[1]
    as.atoms[idx] = tup[2]
    tup
end

"""
    active_set_update!(active_set::ActiveSet, lambda, atom)

Adds the atom to the active set with weight lambda or adds lambda to existing atom.
"""
function active_set_update!(active_set::ActiveSet, lambda, atom)
    # rescale active set
    active_set.weights .*= (1 - lambda)
    # add value for new atom
    idx = find_atom(active_set, atom)
    if idx > 0
        @inbounds active_set.weights[idx] = active_set[idx][1] + lambda
    else
        push!(active_set,(lambda, atom))
    end
    active_set_cleanup!(active_set)
end

function active_set_validate(active_set::ActiveSet)
    return sum(active_set.weights) ≈ 1.0
end

function active_set_renormalize!(active_set::ActiveSet)
    renorm = sum(active_set.weights)
    active_set.weights ./= renorm
    return nothing
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
    compute_active_set_iterate(active_set)
"""
function compute_active_set_iterate(active_set)
    # iteration protocol is obtained for free from
    # the fact that active_set is an abstract vector
    return sum(
        λi * ai for (λi, ai) in active_set
    )
end

function active_set_cleanup!(active_set)
    filter!(e->e[1] > 0, active_set)
end

function find_atom(active_set::ActiveSet, atom)
    for idx in eachindex(active_set)
        if active_set.atoms[idx] == atom
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
    val = Inf
    idx = -1
    temp = 0
    for i in eachindex(active_set)
        temp = LinearAlgebra.dot(active_set.atoms[i], direction)
        if temp < val
            val = temp
            idx = i
        end
    end
    # return lambda, vertex, index
    return (active_set[idx]..., idx)
end    
