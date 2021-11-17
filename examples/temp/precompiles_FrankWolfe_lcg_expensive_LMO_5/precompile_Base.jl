function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(getindex),Dict{Int32, Symbol},Int32})   # time: 0.09366095
    Base.precompile(Tuple{typeof(string),String,BigFloat})   # time: 0.06770378
    Base.precompile(Tuple{typeof(axes),UnitRange{Int64}})   # time: 0.04594267
    Base.precompile(Tuple{typeof(maximum),Vector{Float64}})   # time: 0.013378735
    Base.precompile(Tuple{typeof(minimum),Vector{Float64}})   # time: 0.010232996
    Base.precompile(Tuple{typeof(string),String,Float64,String,Int64,String})   # time: 0.004672595
    Base.precompile(Tuple{typeof(max),Int64,BigFloat})   # time: 0.002166166
    Base.precompile(Tuple{typeof(/),Float64,Int64})   # time: 0.001298473
    Base.precompile(Tuple{typeof(/),UInt64,Float64})   # time: 0.001291698
    Base.precompile(Tuple{typeof(indexed_iterate),Tuple{Union{Float64, Int64, BigFloat}, Union{Nothing, Float64}},Int64})   # time: 0.001064157
    Base.precompile(Tuple{typeof(>),BigFloat,Float64})   # time: 0.001055497
end
