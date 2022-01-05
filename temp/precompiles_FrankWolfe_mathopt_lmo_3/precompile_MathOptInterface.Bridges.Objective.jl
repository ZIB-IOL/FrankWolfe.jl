function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(add_key_for_bridge),Map,SlackBridge,MathOptInterface.ScalarAffineFunction})   # time: 0.017668571
    Base.precompile(Tuple{typeof(add_key_for_bridge),Map,AbstractBridge,MathOptInterface.VariableIndex})   # time: 0.006007369
    Base.precompile(Tuple{typeof(root_bridge),Map})   # time: 0.004581073
    Base.precompile(Tuple{typeof(MathOptInterface.Bridges.added_constraint_types),Core.Type{<:MathOptInterface.Bridges.Objective.SlackBridge{T, F}}})   # time: 0.004175437
    Base.precompile(Tuple{typeof(add_key_for_bridge),Map,AbstractBridge,MathOptInterface.AbstractScalarFunction})   # time: 0.003752136
    Base.precompile(Tuple{Core.Type{MathOptInterface.Bridges.Objective.SlackBridge{T, F<:MathOptInterface.AbstractScalarFunction, G<:MathOptInterface.AbstractScalarFunction}},MathOptInterface.VariableIndex,Any})   # time: 0.003058683
    Base.precompile(Tuple{Core.Type{MathOptInterface.Bridges.Objective.SlackBridge{T, F<:MathOptInterface.AbstractScalarFunction, G<:MathOptInterface.AbstractScalarFunction}},MathOptInterface.VariableIndex,MathOptInterface.ConstraintIndex})   # time: 0.001009342
end
