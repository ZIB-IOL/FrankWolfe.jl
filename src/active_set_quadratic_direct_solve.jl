
"""
    ActiveSetQuadraticLinearSolve{AT, R, IT}

Represents an active set of extreme vertices collected in a FW algorithm,
along with their coefficients `(λ_i, a_i)`.
`R` is the type of the `λ_i`, `AT` is the type of the atoms `a_i`.
The iterate `x = ∑λ_i a_i` is stored in x with type `IT`.
The objective function is assumed to be of the form `f(x)=½⟨x,Ax⟩+⟨b,x⟩+c`
so that the gradient is simply `∇f(x)=Ax+b`.
"""
struct ActiveSetQuadraticLinearSolve{AT, R <: Real, IT, H, OT <: MOI.AbstractOptimizer, AS <: AbstractActiveSet} <: AbstractActiveSet{AT,R,IT}
    weights::Vector{R}
    atoms::Vector{AT}
    x::IT
    A::H # Hessian matrix
    b::IT # linear term
    active_set::AS
    lp_optimizer::OT
    lp_frequency::Int
    counter::Base.RefValue{Int}
end

function ActiveSetQuadraticLinearSolve(tuple_values::AbstractVector{Tuple{R,AT}}, grad!::Function, lp_optimizer; lp_frequency=100) where {AT,R}
    A, b = detect_quadratic_function(grad!, tuple_values[1][2])
    return ActiveSetQuadraticLinearSolve(tuple_values, A, b, lp_optimizer, lp_frequency=lp_frequency)
end

function ActiveSetQuadraticLinearSolve(tuple_values::AbstractVector{Tuple{R,AT}}, A::H, b, lp_optimizer; lp_frequency=10) where {AT,R,H}
    inner_as = ActiveSetQuadratic(tuple_values, A, b)
    as = ActiveSetQuadraticLinearSolve(inner_as.weights, inner_as.atoms, inner_as.x, A, b, inner_as, lp_optimizer, lp_frequency, Ref(0))
    compute_active_set_iterate!(as)
    return as
end

function ActiveSetQuadraticLinearSolve{AT,R}(tuple_values::AbstractVector{<:Tuple{<:Number,<:Any}}, grad!::Function, lp_optimizer; lp_frequency=100) where {AT,R}
    A, b = detect_quadratic_function(grad!, tuple_values[1][2])
    return ActiveSetQuadraticLinearSolve{AT,R}(tuple_values, A, b, lp_optimizer; lp_frequency=lp_frequency)
end

function ActiveSetQuadraticLinearSolve{AT,R}(tuple_values::AbstractVector{<:Tuple{<:Number,<:Any}}, A::H, b, lp_optimizer; lp_frequency=lp_frequency) where {AT,R,H}
    inner_as = ActiveSetQuadratic{AT,R}(tuple_values, A, b)
    as = ActiveSetQuadraticLinearSolve{AT,R,typeof(x),H}(inner_as.weights, inner_as.atoms, inner_as.x, A, b, inner_as, lp_optimizer, lp_frequency, Ref(0))
    compute_active_set_iterate!(as)
    return as
end

function ActiveSetQuadraticLinearSolve(tuple_values::AbstractVector{Tuple{R,AT}}, A::UniformScaling, b, lp_optimizer; lp_frequency=100) where {AT,R}
    return ActiveSetQuadraticLinearSolve(tuple_values, Identity(A.λ), b, lp_optimizer; lp_frequency=lp_frequency)
end
function ActiveSetQuadraticLinearSolve{AT,R}(tuple_values::AbstractVector{<:Tuple{<:Number,<:Any}}, A::UniformScaling, b, lp_optimizer; lp_frequency=100) where {AT,R}
    return ActiveSetQuadraticLinearSolve{AT,R}(tuple_values, Identity(A.λ), b, lp_optimizer; lp_frequency=lp_frequency)
end

# all mutating functions are delegated to the inner active set

Base.push!(as::ActiveSetQuadraticLinearSolve, tuple) = push!(as.active_set, tuple)

Base.deleteat!(as::ActiveSetQuadraticLinearSolve, idx) = deleteat!(as.active_set, idx)

Base.empty!(as::ActiveSetQuadraticLinearSolve) = empty!(as.active_set)

function active_set_update!(
    as::ActiveSetQuadraticLinearSolve{AT,R},
    lambda, atom, renorm=true, idx=nothing;
    weight_purge_threshold=weight_purge_threshold_default(R),
    add_dropped_vertices=false,
    vertex_storage=nothing,
) where {AT,R}
    if idx === nothing
        idx = find_atom(as, atom)
    end
    active_set_update!(as.active_set, lambda, atom, renorm, idx; weight_purge_threshold=weight_purge_threshold, add_dropped_vertices=add_dropped_vertices, vertex_storage=vertex_storage)
    # new atom introduced, we can solve the auxiliary LP
    if idx < 0
        as.counter[] += 1
        if mod(as.counter[], as.lp_frequency) == 0
            solve_quadratic_activeset_lp!(as)
        end
    end
    return as
end

active_set_renormalize!(as::ActiveSetQuadraticLinearSolve) = active_set_renormalize!(as.active_set)

function active_set_argmin(as::ActiveSetQuadraticLinearSolve, direction)
    return active_set_argmin(as.active_set, direction)
end

function active_set_argminmax(as::ActiveSetQuadraticLinearSolve, direction; Φ=0.5)
    return active_set_argminmax(as.active_set, direction; Φ=Φ)
end

# generic quadratic with quadratic information provided
function solve_quadratic_activeset_lp!(as::ActiveSetQuadraticLinearSolve{AT, R, IT, <: AbstractMatrix}) where {AT, R, IT}
    nv = length(as)
    o = as.lp_optimizer
    MOI.empty!(o)
    λ = MOI.add_variables(o, nv)
    # λ ≥ 0, ∑ λ == 1
    MOI.add_constraint.(o, λ, MOI.GreaterThan(0.0))
    MOI.add_constraint(o, sum(λ; init=0.0), MOI.EqualTo(1.0))
    # Aᵗ Q A λ == -Aᵗ b
    for atom in as.atoms
        lhs = MOI.ScalarAffineFunction{Float64}([], 0.0)
        Base.sizehint!(lhs.terms, nv)
        # replaces direct sum because of MOI and MutableArithmetic slow sums
        for j in 1:nv
            # TODO slow
            push!(lhs.terms, MOI.ScalarAffineTerm(dot(atom, as.A, as.atoms[j]), λ[j]))
        end
        rhs = -dot(atom, as.b)
        MOI.add_constraint(o, lhs, MOI.EqualTo(rhs))
    end
    dummy_objective = MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(1.0, λ), 0.0)
    MOI.set(o, MOI.ObjectiveFunction{typeof(dummy_objective)}(), dummy_objective)
    MOI.set(o, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    MOI.optimize!(o)
end

# special case of scaled identity Hessian
function solve_quadratic_activeset_lp!(as::ActiveSetQuadraticLinearSolve{AT, R, IT, <: Identity}) where {AT, R, IT}
    hessian_scaling = as.A.λ
    nv = length(as)
    o = as.lp_optimizer
    MOI.empty!(o)
    λ = MOI.add_variables(o, nv)
    # λ ≥ 0, ∑ λ == 1
    MOI.add_constraint.(o, λ, MOI.GreaterThan(0.0))
    MOI.add_constraint(o, sum(λ; init=0.0), MOI.EqualTo(1.0))
    # a Aᵗ A λ == -Aᵗ b
    for atom in as.atoms
        lhs = MOI.ScalarAffineFunction{Float64}([], 0.0)
        Base.sizehint!(lhs.terms, nv)
        # replaces direct sum because of MOI and MutableArithmetic slow sums
        for j in 1:nv
            push!(lhs.terms, MOI.ScalarAffineTerm(hessian_scaling * dot(atom, as.atoms[j]), λ[j]))
        end
        rhs = -dot(atom, as.b)
        MOI.add_constraint(o, lhs, MOI.EqualTo(rhs))
    end
    dummy_objective = MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(1.0, λ), 0.0)
    MOI.set(o, MOI.ObjectiveFunction{typeof(dummy_objective)}(), dummy_objective)
    MOI.set(o, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    MOI.optimize!(o)
    if MOI.get(o, MOI.TerminationStatus()) in (MOI.OPTIMAL, MOI.FEASIBLE_POINT, MOI.ALMOST_OPTIMAL)
        indices_to_remove = Int[]
        new_weights = R[]
        for idx in eachindex(λ)
            weight_value = MOI.get(o, MOI.VariablePrimal(), λ[idx])
            if weight_value <= 1e-10
                push!(indices_to_remove, idx)
            else
                push!(new_weights, weight_value)
            end
        end
        deleteat!(as.atoms, indices_to_remove)
        deleteat!(as.weights, indices_to_remove)
        @assert length(as) == length(new_weights)
        as.weights .= new_weights
        @assert all(>=(0), new_weights)
        active_set_renormalize!(as)
        compute_active_set_iterate!(as)
    end
    return as
end
