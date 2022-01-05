function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(delete!),Map,MathOptInterface.VariableIndex})   # time: 0.019081527
    Base.precompile(Tuple{typeof(unbridged_map),MathOptInterface.Bridges.Variable.SetMapBridge{T},MathOptInterface.VariableIndex})   # time: 0.015269758
    Base.precompile(Tuple{typeof(register_context),Map,MathOptInterface.ConstraintIndex{MathOptInterface.VectorOfVariables}})   # time: 0.011682957
    Base.precompile(Tuple{typeof(register_context),Map,MathOptInterface.ConstraintIndex{MathOptInterface.VariableIndex}})   # time: 0.00799669
    Base.precompile(Tuple{typeof(function_for),Map,MathOptInterface.ConstraintIndex{MathOptInterface.VectorOfVariables}})   # time: 0.005480958
    Base.precompile(Tuple{typeof(unbridged_function),Map,MathOptInterface.VariableIndex})   # time: 0.005072042
    Base.precompile(Tuple{typeof(register_context),Map,MathOptInterface.ConstraintIndex})   # time: 0.004641476
    Base.precompile(Tuple{typeof(constraint),Map,MathOptInterface.VariableIndex})   # time: 0.001603124
    Base.precompile(Tuple{typeof(unbridged_map),MathOptInterface.Bridges.Variable.VectorizeBridge{T},MathOptInterface.VariableIndex})   # time: 0.001502902
end
