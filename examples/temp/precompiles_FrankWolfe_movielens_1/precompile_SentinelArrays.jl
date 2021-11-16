function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(Base.Broadcast.broadcasted),typeof(big),ChainedVector})   # time: 0.04881463
end
