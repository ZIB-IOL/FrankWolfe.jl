function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(add_key_for_bridge),Map,SlackBridge,MathOptInterface.ScalarAffineFunction{_A} where _A})   # time: 0.023305252
    Base.precompile(Tuple{typeof(empty!),Map})   # time: 0.017667618
    Base.precompile(Tuple{typeof(root_bridge),Map})   # time: 0.007913804
    Base.precompile(Tuple{typeof(add_key_for_bridge),Map,AbstractBridge,MathOptInterface.SingleVariable})   # time: 0.007254789
    Base.precompile(Tuple{Core.Type{MathOptInterface.Bridges.Objective.SlackBridge{T, F<:MathOptInterface.AbstractScalarFunction, G<:MathOptInterface.AbstractScalarFunction}},MathOptInterface.VariableIndex,Any})   # time: 0.005195272
    Base.precompile(Tuple{typeof(add_key_for_bridge),Map,AbstractBridge,MathOptInterface.AbstractScalarFunction})   # time: 0.004613782
    Base.precompile(Tuple{Core.Type{MathOptInterface.Bridges.Objective.SlackBridge{T, F<:MathOptInterface.AbstractScalarFunction, MathOptInterface.SingleVariable}},MathOptInterface.VariableIndex,MathOptInterface.ConstraintIndex{_A, _B} where {_A, _B}})   # time: 0.001675255
end
