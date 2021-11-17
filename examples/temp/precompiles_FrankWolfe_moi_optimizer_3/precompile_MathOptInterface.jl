function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{Type{VectorAffineFunction},Core.Array{MathOptInterface.VectorAffineTerm{T}, 1},Core.Array{T, 1}})   # time: 0.009393923
    Base.precompile(Tuple{typeof(convert),Type{ScalarAffineFunction{_A}} where _A,SingleVariable})   # time: 0.009124899
    Base.precompile(Tuple{Type{VectorQuadraticFunction},Core.Array{MathOptInterface.VectorAffineTerm{T}, 1},Core.Array{MathOptInterface.VectorQuadraticTerm{T}, 1},Core.Array{T, 1}})   # time: 0.006991143
    Base.precompile(Tuple{typeof(get),MathOptInterface.Utilities.UniversalFallback{MathOptInterface.Utilities.GenericModel{Float64, MathOptInterface.Utilities.ModelFunctionConstraints{Float64}}},AbstractConstraintAttribute,Vector{ConstraintIndex{SingleVariable, GreaterThan{Float64}}}})   # time: 0.00615257
    Base.precompile(Tuple{typeof(get),MathOptInterface.Utilities.UniversalFallback{MathOptInterface.Utilities.GenericModel{Float64, MathOptInterface.Utilities.ModelFunctionConstraints{Float64}}},AbstractConstraintAttribute,Vector{ConstraintIndex{ScalarAffineFunction{Float64}, EqualTo{Float64}}}})   # time: 0.00602462
    Base.precompile(Tuple{Type{SetAttributeNotAllowed},ObjectiveFunction})   # time: 0.002557757
    Base.precompile(Tuple{Type{ConstraintIndex{VectorOfVariables, _A}} where _A,Any})   # time: 0.001434238
    Base.precompile(Tuple{Type{VariableIndex},Any})   # time: 0.0013549
end
