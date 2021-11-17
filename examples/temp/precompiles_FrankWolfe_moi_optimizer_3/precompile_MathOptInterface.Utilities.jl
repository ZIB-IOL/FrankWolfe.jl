function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(operate),typeof(-),Core.Type{T<:(MathOptInterface.ScalarAffineFunction{_A} where _A)},T<:(MathOptInterface.ScalarAffineFunction{_A} where _A),MathOptInterface.SingleVariable})   # time: 0.08409089
    Base.precompile(Tuple{typeof(operate_output_index!),typeof(+),Core.Type{T},Int64,MathOptInterface.VectorAffineFunction{T},MathOptInterface.ScalarAffineFunction{T}})   # time: 0.06964898
    Base.precompile(Tuple{typeof(operate_output_index!),typeof(+),Core.Type{T},Int64,MathOptInterface.VectorQuadraticFunction{T},MathOptInterface.ScalarAffineFunction{T}})   # time: 0.06182772
    Base.precompile(Tuple{Core.kwftype(typeof(normalize_constant)),NamedTuple{(:allow_modify_function,), Tuple{Bool}},typeof(normalize_constant),Union{MathOptInterface.ScalarAffineFunction{T}, MathOptInterface.ScalarQuadraticFunction{T}},MathOptInterface.AbstractScalarSet})   # time: 0.028125277
    Base.precompile(Tuple{typeof(operate),typeof(-),Core.Type{T},MathOptInterface.ScalarAffineFunction{T},MathOptInterface.SingleVariable})   # time: 0.011513574
    Base.precompile(Tuple{typeof(operate),typeof(-),Core.Type{T},MathOptInterface.SingleVariable,MathOptInterface.SingleVariable})   # time: 0.010704756
    Base.precompile(Tuple{typeof(operate_output_index!),typeof(+),Core.Type{T<:(MathOptInterface.ScalarAffineFunction{_A} where _A)},Int64,MathOptInterface.VectorAffineFunction{T<:(MathOptInterface.ScalarAffineFunction{_A} where _A)},T<:(MathOptInterface.ScalarAffineFunction{_A} where _A)})   # time: 0.007881186
    Base.precompile(Tuple{typeof(MathOptInterface.get),UniversalFallback{GenericModel{Float64, ModelFunctionConstraints{Float64}}},MathOptInterface.ObjectiveFunction{MathOptInterface.ScalarAffineFunction{Float64}}})   # time: 0.007178028
    Base.precompile(Tuple{typeof(operate!),typeof(+),Type{Float64},MathOptInterface.ScalarAffineFunction{Float64},MathOptInterface.ScalarAffineFunction{Float64}})   # time: 0.003637692
    Base.precompile(Tuple{typeof(operate),typeof(*),Type{Float64},Float64,MathOptInterface.ScalarAffineFunction{Float64}})   # time: 0.003470918
    isdefined(MathOptInterface.Utilities, Symbol("#159#165")) && Base.precompile(Tuple{getfield(MathOptInterface.Utilities, Symbol("#159#165")),DataType})   # time: 0.001908172
    Base.precompile(Tuple{typeof(operate_output_index!),typeof(+),Core.Type{T<:(MathOptInterface.ScalarAffineFunction{_A} where _A)},Int64,MathOptInterface.VectorQuadraticFunction{T<:(MathOptInterface.ScalarAffineFunction{_A} where _A)},T<:(MathOptInterface.ScalarAffineFunction{_A} where _A)})   # time: 0.001795522
    Base.precompile(Tuple{Core.kwftype(typeof(normalize_constant)),NamedTuple{(:allow_modify_function,), Tuple{Bool}},typeof(normalize_constant),MathOptInterface.AbstractScalarFunction,MathOptInterface.AbstractScalarSet})   # time: 0.001728475
    Base.precompile(Tuple{Core.kwftype(typeof(normalize_constant)),NamedTuple{(:allow_modify_function,), Tuple{Bool}},typeof(normalize_constant),MathOptInterface.ScalarAffineFunction{_A} where _A,MathOptInterface.AbstractScalarSet})   # time: 0.001674419
    isdefined(MathOptInterface.Utilities, Symbol("#160#166")) && Base.precompile(Tuple{getfield(MathOptInterface.Utilities, Symbol("#160#166")),DataType})   # time: 0.00164725
    Base.precompile(Tuple{typeof(setindex!),IndexMap,MathOptInterface.ConstraintIndex{MathOptInterface.VectorOfVariables, S},MathOptInterface.ConstraintIndex{MathOptInterface.VectorOfVariables, S}})   # time: 0.001488087
end
