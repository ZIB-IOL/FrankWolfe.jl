function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(getindex),Dict{Int32, Symbol},Int32})   # time: 0.010841742
    Base.precompile(Tuple{typeof(>),BigFloat,Int64})   # time: 0.002152846
    Base.precompile(Tuple{typeof(min),Int64,Float64})   # time: 0.001476575
    Base.precompile(Tuple{typeof(max),Int64,Float64})   # time: 0.001233525
    Base.precompile(Tuple{typeof(/),Float64,Int64})   # time: 0.001207407
    Base.precompile(Tuple{typeof(max),Float64,Int64})   # time: 0.001155028
end
