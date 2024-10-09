
"""
    ActiveSetQuadraticReloaded{AT, R, IT}

Represents an active set of extreme vertices collected in a FW algorithm,
along with their coefficients `(λ_i, a_i)`.
`R` is the type of the `λ_i`, `AT` is the type of the atoms `a_i`.
The iterate `x = ∑λ_i a_i` is stored in x with type `IT`.
The objective function is assumed to be of the form `f(x)=½⟨x,Ax⟩+⟨b,x⟩+c`
so that the gradient is simply `∇f(x)=Ax+b`.
"""
struct ActiveSetQuadraticReloaded{AT, R <: Real, IT, H, OT <: Union{Nothing, MOI.AbstractOptimizer}} <: AbstractActiveSet{AT,R,IT}
    weights::Vector{R}
    atoms::Vector{AT}
    x::IT
    A::H # Hessian matrix
    b::IT # linear term
    lp_optimizer::OT
    lp_frequency::Int
    counter::Base.RefValue{Int}
end

function ActiveSetQuadraticReloaded(tuple_values::AbstractVector{Tuple{R,AT}}, grad!::Function, lp_optimizer=nothing; lp_frequency=10) where {AT,R}
    A, b = detect_quadratic_function(grad!, tuple_values[1][2])
    return ActiveSetQuadraticReloaded(tuple_values, A, b, lp_optimizer, lp_frequency=lp_frequency)
end

function ActiveSetQuadraticReloaded(tuple_values::AbstractVector{Tuple{R,AT}}, A::H, b, lp_optimizer=nothing; lp_frequency=10) where {AT,R,H}
    n = length(tuple_values)
    weights = Vector{R}(undef, n)
    atoms = Vector{AT}(undef, n)
    @inbounds for idx in 1:n
        weights[idx] = tuple_values[idx][1]
        atoms[idx] = tuple_values[idx][2]
    end
    x = similar(b)
    as = ActiveSetQuadraticReloaded(weights, atoms, x, A, b, lp_optimizer, lp_frequency, Ref(0))
    compute_active_set_iterate!(as)
    return as
end

function ActiveSetQuadraticReloaded{AT,R}(tuple_values::AbstractVector{<:Tuple{<:Number,<:Any}}, grad!::Function, lp_optimizer=nothing; lp_frequency=10) where {AT,R}
    A, b = detect_quadratic_function(grad!, tuple_values[1][2])
    return ActiveSetQuadraticReloaded{AT,R}(tuple_values, A, b, lp_optimizer; lp_frequency=lp_frequency)
end

function ActiveSetQuadraticReloaded{AT,R}(tuple_values::AbstractVector{<:Tuple{<:Number,<:Any}}, A::H, b, lp_optimizer=nothing; lp_frequency=lp_frequency) where {AT,R,H}
    n = length(tuple_values)
    weights = Vector{R}(undef, n)
    atoms = Vector{AT}(undef, n)
    @inbounds for idx in 1:n
        weights[idx] = tuple_values[idx][1]
        atoms[idx] = tuple_values[idx][2]
    end
    x = similar(b)
    as = ActiveSetQuadraticReloaded{AT,R,typeof(x),H}(weights, atoms, x, A, b, lp_optimizer, lp_frequency, Ref(0))
    compute_active_set_iterate!(as)
    return as
end

function ActiveSetQuadraticReloaded(tuple_values::AbstractVector{Tuple{R,AT}}, A::UniformScaling, b, lp_optimizer=nothing; lp_frequency=10) where {AT,R}
    return ActiveSetQuadraticReloaded(tuple_values, Identity(A.λ), b, lp_optimizer; lp_frequency=lp_frequency)
end
function ActiveSetQuadraticReloaded{AT,R}(tuple_values::AbstractVector{<:Tuple{<:Number,<:Any}}, A::UniformScaling, b, lp_optimizer=nothing; lp_frequency=10) where {AT,R}
    return ActiveSetQuadraticReloaded{AT,R}(tuple_values, Identity(A.λ), b, lp_optimizer; lp_frequency=lp_frequency)
end

# these three functions do not update the active set iterate

function Base.push!(as::ActiveSetQuadraticReloaded{AT,R}, (λ, a)) where {AT,R}
    push!(as.weights, λ)
    push!(as.atoms, a)
    return as
end

function Base.deleteat!(as::ActiveSetQuadraticReloaded, idx::Int)
    deleteat!(as.weights, idx)
    deleteat!(as.atoms, idx)
    return as
end

function Base.empty!(as::ActiveSetQuadraticReloaded)
    empty!(as.atoms)
    empty!(as.weights)
    as.x .= 0
    return as
end

function active_set_update!(
    active_set::ActiveSetQuadraticReloaded{AT,R},
    lambda, atom, renorm=true, idx=nothing;
    weight_purge_threshold=weight_purge_threshold_default(R),
    add_dropped_vertices=false,
    vertex_storage=nothing,
) where {AT,R}
    # rescale active set
    active_set.weights .*= (1 - lambda)
    # add value for new atom
    if idx === nothing
        idx = find_atom(active_set, atom)
    end
    if idx > 0
        @info "old atom"
        @inbounds active_set.weights[idx] += lambda
    else
        push!(active_set, (lambda, atom))
        if active_set.lp_optimizer !== nothing
            active_set.counter[] += 1
            @show active_set.counter[]
            @show mod(active_set.counter[], active_set.lp_frequency)
            if mod(active_set.counter[], active_set.lp_frequency) == 0
                @show "solving quadratic"
                solve_quadratic_activeset_lp!(active_set)
                return active_set
            end
        end
    end
    if renorm
        add_dropped_vertices = add_dropped_vertices ? vertex_storage !== nothing : add_dropped_vertices
        active_set_cleanup!(active_set; weight_purge_threshold=weight_purge_threshold, update=false, add_dropped_vertices=add_dropped_vertices, vertex_storage=vertex_storage)
        active_set_renormalize!(active_set)
    end
    active_set_update_scale!(active_set.x, lambda, atom)
    return active_set
end

# generic quadratic
function solve_quadratic_activeset_lp!(active_set::ActiveSetQuadraticReloaded{AT, R, IT, H}) where {AT, R, IT, H}
    error("TODO")
end

# special case of scaled identity Hessian
function solve_quadratic_activeset_lp!(active_set::ActiveSetQuadraticReloaded{AT, R, IT, <: Identity}) where {AT, R, IT}
    hessian_scaling = active_set.A.λ
    # number of vertices and ambient dimension
    nv = length(active_set)
    o = active_set.lp_optimizer
    MOI.empty!(o)
    λ = MOI.add_variables(o, nv)
    A = zeros(length(active_set.x), nv)
    for (idx, atom) in enumerate(active_set.atoms)
        A[:,idx] .= atom
    end
    # λ ≥ 0, ∑ λ == 1
    MOI.add_constraint.(o, λ, MOI.GreaterThan(0.0))
    MOI.add_constraint(o, sum(λ; init=0.0), MOI.EqualTo(1.0))
    # 2 * a Aᵗ A λ == -Aᵗ b
    lhs = 0.0 * λ
    rhs = 0 * active_set.weights
    for (idx, atom) in enumerate(active_set.atoms)
        lhs[idx] = 2 * hessian_scaling * sum(λ[j] * dot(atom, active_set.atoms[j]) for j in 1:nv)
        rhs[idx] = -dot(atom, active_set.b)
    end
    MOI.add_constraint.(o, lhs, MOI.EqualTo.(rhs))
    dummy_objective = sum(λ, init=0.0)
    MOI.set(o, MOI.ObjectiveFunction{typeof(dummy_objective)}(), dummy_objective)
    MOI.set(o, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    MOI.optimize!(o)
    if MOI.get(o, MOI.TerminationStatus()) in (MOI.OPTIMAL, MOI.FEASIBLE_POINT, MOI.ALMOST_OPTIMAL)
        indices_to_remove = Int[]
        new_weights = R[]
        for idx in eachindex(λ)
            weight_value = MOI.get(o, MOI.VariablePrimal(), λ[idx])
            if weight_value <= min(1e-3 / nv, 1e-8)
                push!(indices_to_remove, idx)
            else
                push!(new_weights, weight_value)
            end
        end
        deleteat!(active_set.atoms, indices_to_remove)
        deleteat!(active_set.weights, indices_to_remove)
        @assert length(active_set) == length(new_weights)
        active_set.weights .= new_weights
        active_set_renormalize!(active_set)
    end
    return active_set
end

function active_set_renormalize!(active_set::ActiveSetQuadraticReloaded)
    renorm = sum(active_set.weights)
    active_set.weights ./= renorm
    return active_set
end
