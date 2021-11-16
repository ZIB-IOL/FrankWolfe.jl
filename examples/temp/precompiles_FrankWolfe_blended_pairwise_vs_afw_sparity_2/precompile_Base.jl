function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(*),BigFloat,Vector{BigFloat}})   # time: 0.015755232
    Base.precompile(Tuple{typeof(*),Float64,Vector{BigFloat}})   # time: 0.015112486
    Base.precompile(Tuple{typeof(*),BigFloat,Vector{Float64}})   # time: 0.012527041
    Base.precompile(Tuple{typeof(string),String,Type{Nothing},String,Bool,String,Float64,String,Nothing,String,Bool})   # time: 0.008816941
    Base.precompile(Tuple{typeof(getindex),Dict{Int32, Symbol},Int32})   # time: 0.005854781
    Base.precompile(Tuple{typeof(string),Int64})   # time: 0.004800939
    Base.precompile(Tuple{typeof(_array_for),Type{Any},UnitRange{Int64},HasShape{1}})   # time: 0.002085595
    Base.precompile(Tuple{typeof(getindex),Tuple,UnitRange{Int64}})   # time: 0.002074003
    Base.precompile(Tuple{typeof(max),BigFloat,Float64})   # time: 0.001471064
    Base.precompile(Tuple{typeof(axes),UnitRange{Int64}})   # time: 0.001272228
    Base.precompile(Tuple{typeof(min),BigFloat,Int64})   # time: 0.001116756
end
