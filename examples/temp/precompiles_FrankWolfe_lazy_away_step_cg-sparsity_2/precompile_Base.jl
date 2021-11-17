function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(*),Float64,Vector{BigFloat}})   # time: 0.016021106
    Base.precompile(Tuple{typeof(*),BigFloat,Vector{BigFloat}})   # time: 0.015341248
    Base.precompile(Tuple{typeof(*),BigFloat,Vector{Float64}})   # time: 0.013977899
    Base.precompile(Tuple{typeof(string),String,Type{Nothing},String,Bool,String,Float64,String,Nothing,String,Bool})   # time: 0.008168175
    Base.precompile(Tuple{typeof(getindex),Dict{Int32, Symbol},Int32})   # time: 0.005847849
    Base.precompile(Tuple{typeof(string),Int64})   # time: 0.004558801
    Base.precompile(Tuple{typeof(getindex),Tuple,UnitRange{Int64}})   # time: 0.001909825
    Base.precompile(Tuple{typeof(_array_for),Type{Any},UnitRange{Int64},HasShape{1}})   # time: 0.001736218
    Base.precompile(Tuple{typeof(max),BigFloat,Float64})   # time: 0.001464268
    Base.precompile(Tuple{typeof(axes),UnitRange{Int64}})   # time: 0.001305462
end
