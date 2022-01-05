function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(_broadcast_getindex_evalf),typeof(*),Int64,Float64})   # time: 0.001269216
end
