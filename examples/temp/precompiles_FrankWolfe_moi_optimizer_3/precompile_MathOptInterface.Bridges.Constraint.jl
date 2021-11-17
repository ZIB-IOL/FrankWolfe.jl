function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(add_key_for_bridge),Map,AbstractBridge,MathOptInterface.SingleVariable,MathOptInterface.AbstractScalarSet})   # time: 0.058707453
    Base.precompile(Tuple{typeof(variable_constraints),Map,MathOptInterface.VariableIndex})   # time: 0.013401079
    Base.precompile(Tuple{typeof(empty!),Map})   # time: 0.013228227
    Base.precompile(Tuple{typeof(haskey),Map,MathOptInterface.ConstraintIndex{MathOptInterface.SingleVariable, MathOptInterface.LessThan{T}}})   # time: 0.008743505
    Base.precompile(Tuple{typeof(haskey),Map,MathOptInterface.ConstraintIndex{MathOptInterface.SingleVariable, MathOptInterface.GreaterThan{T}}})   # time: 0.008407221
    Base.precompile(Tuple{typeof(add_key_for_bridge),Map,AbstractBridge,MathOptInterface.VectorOfVariables,MathOptInterface.AbstractVectorSet})   # time: 0.00539634
    Base.precompile(Tuple{typeof(haskey),Map,MathOptInterface.ConstraintIndex{F, MathOptInterface.LessThan{T}} where {T, F<:MathOptInterface.SingleVariable}})   # time: 0.005304499
    Base.precompile(Tuple{typeof(haskey),Map,MathOptInterface.ConstraintIndex{F, MathOptInterface.GreaterThan{T}} where {T, F<:MathOptInterface.SingleVariable}})   # time: 0.004792156
    Base.precompile(Tuple{typeof(haskey),Map,MathOptInterface.ConstraintIndex{MathOptInterface.SingleVariable, _A} where _A})   # time: 0.004463556
    Base.precompile(Tuple{typeof(haskey),Map,Union{MathOptInterface.ConstraintIndex{F, MathOptInterface.GreaterThan{T}}, MathOptInterface.ConstraintIndex{F, MathOptInterface.LessThan{T}}} where {T, F<:MathOptInterface.AbstractScalarFunction}})   # time: 0.004336171
    Base.precompile(Tuple{typeof(add_key_for_bridge),Map,AbstractBridge,MathOptInterface.SingleVariable,MathOptInterface.GreaterThan{Float64}})   # time: 0.003466243
    Base.precompile(Tuple{typeof(vector_of_variables_constraints),Map})   # time: 0.003153846
    Base.precompile(Tuple{typeof(delete!),Map,Union{MathOptInterface.ConstraintIndex{F, MathOptInterface.GreaterThan{T}}, MathOptInterface.ConstraintIndex{F, MathOptInterface.LessThan{T}}} where {T, F<:MathOptInterface.AbstractScalarFunction}})   # time: 0.002825875
    Base.precompile(Tuple{typeof(delete!),Map,Union{MathOptInterface.ConstraintIndex{MathOptInterface.SingleVariable, MathOptInterface.GreaterThan{T}}, MathOptInterface.ConstraintIndex{MathOptInterface.SingleVariable, MathOptInterface.LessThan{T}}}})   # time: 0.002149401
    Base.precompile(Tuple{typeof(haskey),Map,MathOptInterface.ConstraintIndex{MathOptInterface.VectorOfVariables, S} where S})   # time: 0.002146259
    Base.precompile(Tuple{typeof(haskey),Map,MathOptInterface.ConstraintIndex{MathOptInterface.SingleVariable, MathOptInterface.GreaterThan{Float64}}})   # time: 0.001634329
    Base.precompile(Tuple{typeof(add_key_for_bridge),Map,AbstractBridge,MathOptInterface.ScalarAffineFunction{_A} where _A,MathOptInterface.AbstractSet})   # time: 0.001353781
    Base.precompile(Tuple{typeof(getindex),Map,Union{MathOptInterface.ConstraintIndex{MathOptInterface.SingleVariable, MathOptInterface.GreaterThan{T}}, MathOptInterface.ConstraintIndex{MathOptInterface.SingleVariable, MathOptInterface.LessThan{T}}}})   # time: 0.001059895
end
