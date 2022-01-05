function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(getindex),Dict{Int32, Symbol},Int32})   # time: 0.009244422
    Base.precompile(Tuple{typeof(min),Int64,Float64})   # time: 0.00163567
    Base.precompile(Tuple{typeof(max),Int64,Float64})   # time: 0.001455343
    Base.precompile(Tuple{typeof(>),BigFloat,Int64})   # time: 0.001371046
    Base.precompile(Tuple{typeof(/),Float64,Int64})   # time: 0.001312936
    Base.precompile(Tuple{typeof(_similar_shape),UnitRange{Int64},HasShape{1}})   # time: 0.001079884
    Base.precompile(Tuple{typeof(/),UInt64,Float64})   # time: 0.001015207
end
