function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(broadcasted),typeof(*),Int64,UniformScaling})   # time: 0.003213835
end
