function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(add_key_for_bridge),Map,SlackBridge,MathOptInterface.ScalarAffineFunction})   # time: 0.07842007
    Base.precompile(Tuple{typeof(add_key_for_bridge),Map,AbstractBridge,MathOptInterface.VariableIndex})   # time: 0.009960471
    Base.precompile(Tuple{typeof(root_bridge),Map})   # time: 0.009943987
    Base.precompile(Tuple{typeof(add_key_for_bridge),Map,AbstractBridge,MathOptInterface.AbstractScalarFunction})   # time: 0.006039545
    Base.precompile(Tuple{typeof(MathOptInterface.Bridges.added_constraint_types),Core.Type{<:MathOptInterface.Bridges.Objective.SlackBridge{T, F}}})   # time: 0.006014263
    Base.precompile(Tuple{Core.Type{MathOptInterface.Bridges.Objective.SlackBridge{T, F<:MathOptInterface.AbstractScalarFunction, G<:MathOptInterface.AbstractScalarFunction}},MathOptInterface.VariableIndex,Any})   # time: 0.004711105
    Base.precompile(Tuple{Core.Type{MathOptInterface.Bridges.Objective.SlackBridge{T, F<:MathOptInterface.AbstractScalarFunction, G<:MathOptInterface.AbstractScalarFunction}},MathOptInterface.VariableIndex,MathOptInterface.ConstraintIndex})   # time: 0.001764385
    Base.precompile(Tuple{typeof(concrete_bridge_type),MathOptInterface.Bridges.AbstractBridgeOptimizer,Type{<:MathOptInterface.AbstractScalarFunction}})   # time: 0.001056479
end
