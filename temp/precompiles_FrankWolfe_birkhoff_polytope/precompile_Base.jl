function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(getindex),Dict{Int32, Symbol},Int32})   # time: 0.005860418
    Base.precompile(Tuple{typeof(/),Float64,Int64})   # time: 0.001594959
    Base.precompile(Tuple{typeof(_similar_shape),UnitRange{Int64},HasShape{1}})   # time: 0.001287169
    Base.precompile(Tuple{typeof(max),Int64,Float64})   # time: 0.001193931
    Base.precompile(Tuple{typeof(>),BigFloat,Int64})   # time: 0.001083305
end
