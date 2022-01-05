function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(delete!),Map,MathOptInterface.VariableIndex})   # time: 0.010239237
    Base.precompile(Tuple{typeof(unbridged_map),MathOptInterface.Bridges.Variable.SetMapBridge{T},MathOptInterface.VariableIndex})   # time: 0.010038935
    Base.precompile(Tuple{typeof(register_context),Map,MathOptInterface.ConstraintIndex{MathOptInterface.VariableIndex}})   # time: 0.00508442
    Base.precompile(Tuple{typeof(register_context),Map,MathOptInterface.ConstraintIndex{MathOptInterface.VectorOfVariables}})   # time: 0.004075214
    Base.precompile(Tuple{typeof(register_context),Map,MathOptInterface.ConstraintIndex})   # time: 0.002781067
    Base.precompile(Tuple{typeof(unbridged_function),Map,MathOptInterface.VariableIndex})   # time: 0.001989351
    Base.precompile(Tuple{typeof(function_for),Map,MathOptInterface.ConstraintIndex{MathOptInterface.VectorOfVariables}})   # time: 0.001863286
    Base.precompile(Tuple{typeof(constraint),Map,MathOptInterface.VariableIndex})   # time: 0.001398774
end
