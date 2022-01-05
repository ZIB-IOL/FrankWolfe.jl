function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(add_key_for_bridge),Map,AbstractBridge,MathOptInterface.VariableIndex,MathOptInterface.AbstractScalarSet})   # time: 0.018456526
    Base.precompile(Tuple{typeof(variable_constraints),Map,MathOptInterface.VariableIndex})   # time: 0.016591424
    Base.precompile(Tuple{typeof(add_key_for_bridge),Map,AbstractBridge,MathOptInterface.VectorOfVariables,MathOptInterface.AbstractVectorSet})   # time: 0.012363274
    Base.precompile(Tuple{typeof(haskey),Map,MathOptInterface.ConstraintIndex{MathOptInterface.VariableIndex, MathOptInterface.GreaterThan{T}}})   # time: 0.011876131
    Base.precompile(Tuple{typeof(haskey),Map,MathOptInterface.ConstraintIndex{MathOptInterface.VariableIndex, MathOptInterface.LessThan{T}}})   # time: 0.011486837
    Base.precompile(Tuple{typeof(haskey),Map,MathOptInterface.ConstraintIndex{F, MathOptInterface.GreaterThan{T}} where {T, F<:MathOptInterface.VariableIndex}})   # time: 0.007128634
    Base.precompile(Tuple{typeof(haskey),Map,Union{MathOptInterface.ConstraintIndex{MathOptInterface.VariableIndex, MathOptInterface.GreaterThan{T}}, MathOptInterface.ConstraintIndex{MathOptInterface.VariableIndex, MathOptInterface.LessThan{T}}} where T})   # time: 0.006526827
    Base.precompile(Tuple{typeof(haskey),Map,Union{MathOptInterface.ConstraintIndex{F, MathOptInterface.GreaterThan{T}}, MathOptInterface.ConstraintIndex{F, MathOptInterface.LessThan{T}}} where {T, F<:MathOptInterface.AbstractScalarFunction}})   # time: 0.006422829
    Base.precompile(Tuple{typeof(haskey),Map,MathOptInterface.ConstraintIndex{MathOptInterface.VectorAffineFunction{T}} where T})   # time: 0.006172382
    Base.precompile(Tuple{typeof(vector_of_variables_constraints),Map})   # time: 0.006142637
    Base.precompile(Tuple{typeof(haskey),Map,MathOptInterface.ConstraintIndex{F, MathOptInterface.LessThan{T}} where {T, F<:MathOptInterface.VariableIndex}})   # time: 0.006018452
    Base.precompile(Tuple{typeof(haskey),Map,MathOptInterface.ConstraintIndex{<:Union{MathOptInterface.VariableIndex, MathOptInterface.VectorOfVariables}}})   # time: 0.005745798
    Base.precompile(Tuple{typeof(delete!),Map,Union{MathOptInterface.ConstraintIndex{MathOptInterface.VariableIndex, MathOptInterface.GreaterThan{T}}, MathOptInterface.ConstraintIndex{MathOptInterface.VariableIndex, MathOptInterface.LessThan{T}}}})   # time: 0.004078068
    Base.precompile(Tuple{typeof(delete!),Map,Union{MathOptInterface.ConstraintIndex{F, MathOptInterface.GreaterThan{T}}, MathOptInterface.ConstraintIndex{F, MathOptInterface.LessThan{T}}} where {T, F<:MathOptInterface.AbstractScalarFunction}})   # time: 0.003261841
    Base.precompile(Tuple{typeof(haskey),Map,MathOptInterface.ConstraintIndex{MathOptInterface.VectorOfVariables}})   # time: 0.003141365
    Base.precompile(Tuple{typeof(haskey),Map,MathOptInterface.ConstraintIndex})   # time: 0.00278064
    Base.precompile(Tuple{typeof(add_key_for_bridge),Map,AbstractBridge,MathOptInterface.VectorOfVariables,MathOptInterface.AbstractSet})   # time: 0.002153468
    Base.precompile(Tuple{typeof(add_key_for_bridge),Map,AbstractBridge,MathOptInterface.VariableIndex,MathOptInterface.AbstractSet})   # time: 0.002106814
    Base.precompile(Tuple{typeof(add_key_for_bridge),Map,AbstractBridge,MathOptInterface.ScalarAffineFunction,MathOptInterface.AbstractSet})   # time: 0.002085794
    Base.precompile(Tuple{typeof(add_key_for_bridge),Map,AbstractBridge,MathOptInterface.AbstractFunction,MathOptInterface.AbstractSet})   # time: 0.002078949
    Base.precompile(Tuple{typeof(getindex),Map,MathOptInterface.ConstraintIndex{MathOptInterface.VariableIndex, S}})   # time: 0.001307698
    Base.precompile(Tuple{typeof(getindex),Map,Union{MathOptInterface.ConstraintIndex{MathOptInterface.VariableIndex, MathOptInterface.GreaterThan{T}}, MathOptInterface.ConstraintIndex{MathOptInterface.VariableIndex, MathOptInterface.LessThan{T}}}})   # time: 0.001167527
end
