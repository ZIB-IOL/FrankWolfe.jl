function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(<),BigFloat,Int64})   # time: 0.001291777
end
