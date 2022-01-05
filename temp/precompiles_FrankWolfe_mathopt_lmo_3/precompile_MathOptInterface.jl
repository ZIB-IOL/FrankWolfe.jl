function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{Type{VectorQuadraticFunction},Core.Array{<:MathOptInterface.VectorAffineTerm{T}, 1},Core.Array{<:MathOptInterface.VectorQuadraticTerm{T}, 1},Core.Array{T, 1}})   # time: 0.008370645
    Base.precompile(Tuple{Type{VectorAffineFunction},Core.Array{MathOptInterface.VectorAffineTerm{T}, 1},Core.Array{T, 1}})   # time: 0.005455067
    Base.precompile(Tuple{typeof(convert),Type{ScalarAffineFunction{_A}} where _A,VariableIndex})   # time: 0.004108463
    Base.precompile(Tuple{Type{SetAttributeNotAllowed},ObjectiveFunction})   # time: 0.00223533
    Base.precompile(Tuple{Type{SetAttributeNotAllowed},AbstractVariableAttribute})   # time: 0.002034955
    Base.precompile(Tuple{Type{LessThan},T<:Core.Real})   # time: 0.001729751
    Base.precompile(Tuple{typeof(update_dimension),Union{Nonnegatives, Nonpositives, Reals, Zeros},Any})   # time: 0.001655229
    Base.precompile(Tuple{Type{GreaterThan},T<:Core.Real})   # time: 0.001409391
end
