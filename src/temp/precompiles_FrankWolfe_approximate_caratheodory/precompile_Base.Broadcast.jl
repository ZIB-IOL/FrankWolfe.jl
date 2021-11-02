function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(_broadcast_getindex_evalf),typeof(*),Int64,Rational{BigInt}})   # time: 0.001930476
end
