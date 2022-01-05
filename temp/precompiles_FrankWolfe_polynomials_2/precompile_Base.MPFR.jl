function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(sum),Vector{BigFloat}})   # time: 0.00419178
    Base.precompile(Tuple{typeof(<),BigFloat,Int64})   # time: 0.002046022
    Base.precompile(Tuple{typeof(==),BigFloat,Float64})   # time: 0.001087511
end
