function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{Type{VectorQuadraticFunction},Core.Array{<:MathOptInterface.VectorAffineTerm{T}, 1},Core.Array{<:MathOptInterface.VectorQuadraticTerm{T}, 1},Core.Array{T, 1}})   # time: 0.019283228
    Base.precompile(Tuple{Type{VectorAffineFunction},Core.Array{MathOptInterface.VectorAffineTerm{T}, 1},Core.Array{T, 1}})   # time: 0.01183807
    Base.precompile(Tuple{typeof(convert),Type{ScalarAffineFunction{_A}} where _A,VariableIndex})   # time: 0.00644895
    Base.precompile(Tuple{Type{SetAttributeNotAllowed},AbstractVariableAttribute})   # time: 0.00323188
    Base.precompile(Tuple{Type{SetAttributeNotAllowed},ObjectiveFunction})   # time: 0.003170558
    Base.precompile(Tuple{Type{LessThan},T<:Core.Real})   # time: 0.002401264
    Base.precompile(Tuple{typeof(update_dimension),Union{Nonnegatives, Nonpositives, Reals, Zeros},Any})   # time: 0.002359638
    Base.precompile(Tuple{Type{GreaterThan},T<:Core.Real})   # time: 0.002289377
    Base.precompile(Tuple{Type{AddConstraintNotAllowed{VectorOfVariables, _A}} where _A})   # time: 0.001292961
    Base.precompile(Tuple{Type{AddConstraintNotAllowed{_A, _B}} where {_A, _B}})   # time: 0.001147134
end
