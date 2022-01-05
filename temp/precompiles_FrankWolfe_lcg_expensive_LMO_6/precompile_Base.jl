function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(getindex),Dict{Int32, Symbol},Int32})   # time: 0.013677901
    Base.precompile(Tuple{typeof(>),BigFloat,Int64})   # time: 0.002323044
    Base.precompile(Tuple{typeof(min),Int64,Float64})   # time: 0.00216283
    Base.precompile(Tuple{typeof(max),Int64,Float64})   # time: 0.001887304
    Base.precompile(Tuple{typeof(max),Float64,Int64})   # time: 0.001757964
    Base.precompile(Tuple{typeof(_similar_shape),UnitRange{Int64},HasShape{1}})   # time: 0.001556896
    Base.precompile(Tuple{typeof(/),Float64,Int64})   # time: 0.001445632
    Base.precompile(Tuple{typeof(/),UInt64,Float64})   # time: 0.001278156
end
