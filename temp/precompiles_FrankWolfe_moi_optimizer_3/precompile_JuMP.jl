function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{Type{MathOptInterface.VectorOfVariables},Vector{VariableRef}})   # time: 0.032715198
end
