function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(MathOptInterface.get),Optimizer,MathOptInterface.ListOfVariableIndices})   # time: 0.055345323
end
