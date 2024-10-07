struct ActiveSetSparsifier{AT, R, IT, AS <: AbstractActiveSet{AT, R, IT}, OT <: MOI.AbstractOptimizer} <: AbstractActiveSet{AT,R,IT}
    active_set::AS
    weights::Vector{R}
    atoms::Vector{AT}
    x::IT
    optimizer::OT
    minimum_vertices::Int
    solve_frequency::Int
    counter::Base.RefValue{Int}
end

function ActiveSetSparsifier(active_set::AbstractActiveSet, optimizer::MOI.AbstractOptimizer; minimum_vertices=50, solve_frequency=50)
    return ActiveSetSparsifier(active_set, active_set.weights, active_set.atoms, active_set.x, optimizer, minimum_vertices, solve_frequency, Ref(0))
end

function Base.push!(as::ActiveSetSparsifier, (λ, a))
    push!(as.active_set, (λ, a))
end

Base.deleteat!(as::ActiveSetSparsifier, idx::Int) = deleteat!(as.active_set, idx)

Base.empty!(as::ActiveSetSparsifier) = empty!(as.active_set)

function active_set_update!(
    as::ActiveSetSparsifier{AS, OT, AT, R, IT},
    lambda, atom, renorm=true, idx=nothing;
    kwargs...
) where {AS, OT, AT, R, IT}
    active_set_update!(as.active_set, lambda, atom, renorm, idx; kwargs...)
    n = length(as)
    as.counter[] += 1
    if n > as.minimum_vertices && mod(as.counter[], as.solve_frequency) == 0
        # sparsifying active set
        MOI.empty!(as.optimizer)
        x0 = as.active_set.x
        λ = MOI.add_variables(as.optimizer, n)
        # λ ∈ Δ_n ⇔ λ ≥ 0, ∑ λ == 1
        MOI.add_constraint.(as.optimizer, λ, MOI.GreaterThan(0.0))
        MOI.add_constraint(as.optimizer, sum(λ; init=0.0), MOI.EqualTo(1.0))
        x_sum = 0 * as.active_set.atoms[1]
        for (idx, atom) in enumerate(as.active_set.atoms)
            x_sum += λ[idx] * as.active_set.atoms
        end
        for idx in eachindex(x_sum)
            MOI.add_constraint(as.optimizer, x_sum[idx], MOI.EqualTo(x0[idx]))
        end
        # Set a dummy objective (minimize ∑λ)
        dummy_objective = sum(λ; init=0.0)
        MOI.set(as.optimizer, MOI.ObjectiveFunction{typeof(dummy_objective)}(), dummy_objective)
        MOI.set(as.optimizer, MOI.ObjectiveSense(), MOI.MIN_SENSE)
        MOI.optimize!(as.optimizer)
        if MOI.get(as.optimizer, MOI.TerminationStatus()) in (MOI.OPTIMAL, MOI.FEASIBLE_POINT, MOI.ALMOST_OPTIMAL)
            indices_to_remove = Int[]
            new_weights = R[]
            for idx in eachindex(λ)
                weight_value = MOI.get(as.optimizer, MOI.VariablePrimal(), λ[idx])
                if weight_value <= min(1e-3 / n, 1e-8)
                    push!(indices_to_remove, idx)
                else
                    push!(new_weights, weight_value)
                end
            end
            deleteat!(as.active_set.atoms, indices_to_remove)
            deleteat!(as.active_set.weights, indices_to_remove)
            @assert length(as) == length(new_weights)
            as.active_set.weights .= new_weights
            active_set_renormalize!(as)
        end
    end
    return as
end

active_set_renormalize!(as::ActiveSetSparsifier) = active_set_renormalize!(as.active_set)

active_set_argmin(as::ActiveSetSparsifier, direction) = active_set_argmin(as.active_set, direction)
active_set_argminmax(as::ActiveSetSparsifier, direction; Φ=0.5) = active_set_argminmax(as.active_set, direction; Φ=Φ)
