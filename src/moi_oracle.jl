
"""
    MathOptLMO{OT <: MOI.Optimizer} <: LinearMinimizationOracle

Linear minimization oracle with feasible space defined through a MathOptInterface.Optimizer.
The oracle call sets the direction and reruns the optimizer.


The `direction` vector has to be set in the same order of variables as the `MathOptInterface.ListOfVariableIndices()` getter.
"""
struct MathOptLMO{OT <: MOI.AbstractOptimizer} <: LinearMinimizationOracle
    o::OT
end

function compute_extreme_point(lmo::MathOptLMO{OT}, direction::AbstractVector{T}) where {OT, T <: Real}
    variables = MOI.get(lmo.o, MathOptInterface.ListOfVariableIndices())
    obj = MathOptInterface.ScalarAffineFunction(
        MOI.ScalarAffineTerm.(direction, variables),
        zero(T),
    )
    MOI.set(lmo.o, MOI.ObjectiveFunction{typeof(obj)}(), obj)
    MOI.set(lmo.o, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    _optimize_and_return(lmo)
end

function compute_extreme_point(lmo::MathOptLMO{OT}, direction::AbstractVector{MOI.ScalarAffineTerm{T}}) where {OT, T}
    variables = MOI.get(lmo.o, MathOptInterface.ListOfVariableIndices())
    obj = MathOptInterface.ScalarAffineFunction(
        direction,
        zero(T),
    )
    MOI.set(lmo.o, MOI.ObjectiveFunction{typeof(obj)}(), obj)
    MOI.set(lmo.o, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    return _optimize_and_return(lmo)
end

function _optimize_and_return(lmo)
    MOI.optimize!(lmo.o)
    term_st = MOI.get(lmo.o, MathOptInterface.TerminationStatus())
    if term_st ∉ (MOI.OPTIMAL, MathOptInterface.ALMOST_OPTIMAL)
        @error "Unexpected termionation: $term_st"
        return nothing
    end
    variables = MOI.get(lmo.o, MathOptInterface.ListOfVariableIndices())
    return MOI.get.(lmo.o, MathOptInterface.VariablePrimal(), variables)
end
