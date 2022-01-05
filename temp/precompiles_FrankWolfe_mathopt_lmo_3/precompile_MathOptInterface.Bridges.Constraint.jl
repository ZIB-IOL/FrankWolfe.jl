function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(add_key_for_bridge),Map,AbstractBridge,MathOptInterface.VariableIndex,MathOptInterface.AbstractScalarSet})   # time: 0.010755294
    Base.precompile(Tuple{typeof(variable_constraints),Map,MathOptInterface.VariableIndex})   # time: 0.009995008
    Base.precompile(Tuple{typeof(haskey),Map,MathOptInterface.ConstraintIndex{MathOptInterface.VariableIndex, MathOptInterface.GreaterThan{T}}})   # time: 0.006872138
    Base.precompile(Tuple{typeof(haskey),Map,MathOptInterface.ConstraintIndex{MathOptInterface.VariableIndex, MathOptInterface.LessThan{T}}})   # time: 0.004688704
    Base.precompile(Tuple{typeof(add_key_for_bridge),Map,AbstractBridge,MathOptInterface.VectorOfVariables,MathOptInterface.AbstractVectorSet})   # time: 0.004531767
    Base.precompile(Tuple{typeof(haskey),Map,Union{MathOptInterface.ConstraintIndex{F, MathOptInterface.GreaterThan{T}}, MathOptInterface.ConstraintIndex{F, MathOptInterface.LessThan{T}}} where {T, F<:MathOptInterface.AbstractScalarFunction}})   # time: 0.002861454
    Base.precompile(Tuple{typeof(haskey),Map,Union{MathOptInterface.ConstraintIndex{MathOptInterface.VariableIndex, MathOptInterface.GreaterThan{T}}, MathOptInterface.ConstraintIndex{MathOptInterface.VariableIndex, MathOptInterface.LessThan{T}}} where T})   # time: 0.002759003
    Base.precompile(Tuple{typeof(haskey),Map,MathOptInterface.ConstraintIndex{F, MathOptInterface.LessThan{T}} where {T, F<:MathOptInterface.VariableIndex}})   # time: 0.002615406
    Base.precompile(Tuple{typeof(haskey),Map,MathOptInterface.ConstraintIndex{F, MathOptInterface.GreaterThan{T}} where {T, F<:MathOptInterface.VariableIndex}})   # time: 0.002601434
    Base.precompile(Tuple{typeof(haskey),Map,MathOptInterface.ConstraintIndex{<:Union{MathOptInterface.VariableIndex, MathOptInterface.VectorOfVariables}}})   # time: 0.002369563
    Base.precompile(Tuple{typeof(vector_of_variables_constraints),Map})   # time: 0.002165837
    Base.precompile(Tuple{typeof(haskey),Map,MathOptInterface.ConstraintIndex{MathOptInterface.VectorAffineFunction{T}} where T})   # time: 0.002096903
    Base.precompile(Tuple{typeof(delete!),Map,Union{MathOptInterface.ConstraintIndex{F, MathOptInterface.GreaterThan{T}}, MathOptInterface.ConstraintIndex{F, MathOptInterface.LessThan{T}}} where {T, F<:MathOptInterface.AbstractScalarFunction}})   # time: 0.001478464
    Base.precompile(Tuple{typeof(add_key_for_bridge),Map,AbstractBridge,MathOptInterface.VectorOfVariables,MathOptInterface.AbstractSet})   # time: 0.001437467
    Base.precompile(Tuple{typeof(add_key_for_bridge),Map,AbstractBridge,MathOptInterface.ScalarAffineFunction,MathOptInterface.AbstractSet})   # time: 0.001386343
    Base.precompile(Tuple{typeof(add_key_for_bridge),Map,AbstractBridge,MathOptInterface.VariableIndex,MathOptInterface.AbstractSet})   # time: 0.001375448
    Base.precompile(Tuple{typeof(add_key_for_bridge),Map,AbstractBridge,MathOptInterface.AbstractFunction,MathOptInterface.AbstractSet})   # time: 0.001364482
    Base.precompile(Tuple{typeof(haskey),Map,MathOptInterface.ConstraintIndex{MathOptInterface.VectorOfVariables}})   # time: 0.001339617
    Base.precompile(Tuple{typeof(haskey),Map,MathOptInterface.ConstraintIndex})   # time: 0.001259721
    Base.precompile(Tuple{typeof(delete!),Map,Union{MathOptInterface.ConstraintIndex{MathOptInterface.VariableIndex, MathOptInterface.GreaterThan{T}}, MathOptInterface.ConstraintIndex{MathOptInterface.VariableIndex, MathOptInterface.LessThan{T}}}})   # time: 0.001245058
end
