function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(register_context),Map,MathOptInterface.ConstraintIndex{MathOptInterface.SingleVariable, MathOptInterface.GreaterThan{Float64}}})   # time: 0.038799927
    Base.precompile(Tuple{typeof(register_context),Map,MathOptInterface.ConstraintIndex{MathOptInterface.VectorOfVariables, _A} where _A})   # time: 0.037540007
    Base.precompile(Tuple{typeof(empty!),Map})   # time: 0.026551776
    Base.precompile(Tuple{typeof(register_context),Map,MathOptInterface.ConstraintIndex{MathOptInterface.SingleVariable, _A} where _A})   # time: 0.008723603
    Base.precompile(Tuple{typeof(delete!),Map,MathOptInterface.VariableIndex})   # time: 0.007510474
    Base.precompile(Tuple{typeof(unbridged_function),Map,MathOptInterface.VariableIndex})   # time: 0.004191457
    Base.precompile(Tuple{typeof(register_context),Map,MathOptInterface.ConstraintIndex{_A, _B} where {_A, _B}})   # time: 0.003606408
    Base.precompile(Tuple{typeof(unbridged_map),MathOptInterface.Bridges.Variable.VectorizeBridge{T, S} where S,MathOptInterface.VariableIndex})   # time: 0.00328228
    Base.precompile(Tuple{typeof(function_for),Map,MathOptInterface.ConstraintIndex{MathOptInterface.VectorOfVariables, S} where S})   # time: 0.002822799
    Base.precompile(Tuple{typeof(constraint),Map,MathOptInterface.VariableIndex})   # time: 0.001246088
    Base.precompile(Tuple{typeof(haskey),Map,MathOptInterface.VariableIndex})   # time: 0.001047394
end
