function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(Base.Broadcast.broadcastable),VariablePrimal})   # time: 0.001096429
end
