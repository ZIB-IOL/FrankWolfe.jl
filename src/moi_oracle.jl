
"""
    MathOptLMO{OT <: MOI.Optimizer} <: LinearMinimizationOracle

Linear minimization oracle with feasible space defined through a MathOptInterface.Optimizer.
The oracle call sets the direction and reruns the optimizer.

The `direction` vector has to be set in the same order of variables as the `MOI.ListOfVariableIndices()` getter.
"""
struct MathOptLMO{OT<:MOI.AbstractOptimizer} <: LinearMinimizationOracle
    o::OT
end

function compute_extreme_point(lmo::MathOptLMO{OT}, direction::AbstractVector{T}; kwargs...) where {OT,T<:Real}
    variables = MOI.get(lmo.o, MOI.ListOfVariableIndices())
    terms = [MOI.ScalarAffineTerm(d, v) for (d, v) in zip(direction, variables)]
    obj = MOI.ScalarAffineFunction(terms, zero(T))
    MOI.set(lmo.o, MOI.ObjectiveFunction{typeof(obj)}(), obj)
    MOI.set(lmo.o, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    return _optimize_and_return(lmo, variables)
end

function compute_extreme_point(lmo::MathOptLMO{OT}, direction::AbstractMatrix{T}; kwargs...) where {OT,T<:Real}
    n = size(direction, 1)
    v = compute_extreme_point(lmo, vec(direction))
    return reshape(v, n, n)
end

function compute_extreme_point(
    lmo::MathOptLMO{OT},
    direction::AbstractVector{MOI.ScalarAffineTerm{T}};
    kwargs...
) where {OT,T}
    variables = [term.variable_index for term in direction]
    obj = MOI.ScalarAffineFunction(direction, zero(T))
    MOI.set(lmo.o, MOI.ObjectiveFunction{typeof(obj)}(), obj)
    MOI.set(lmo.o, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    return _optimize_and_return(lmo, variables)
end

function _optimize_and_return(lmo, variables)
    MOI.optimize!(lmo.o)
    term_st = MOI.get(lmo.o, MOI.TerminationStatus())
    if term_st ∉ (MOI.OPTIMAL, MOI.ALMOST_OPTIMAL)
        @error "Unexpected termionation: $term_st"
        return nothing
    end
    return MOI.get.(lmo.o, MOI.VariablePrimal(), variables)
end

"""
    convert_mathopt(lmo::LMO, optimizer::OT; kwargs...) -> MathOptLMO{OT}

Converts the given LMO to its equivalent MathOptInterface representation using `optimizer`.
Must be implemented by LMOs.
"""
function convert_mathopt end
