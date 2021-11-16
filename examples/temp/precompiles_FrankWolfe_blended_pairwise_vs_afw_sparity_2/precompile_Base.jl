function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(getindex),Dict{Int32, Symbol},Int32})   # time: 0.029624362
    Base.precompile(Tuple{typeof(*),Float64,Vector{BigFloat}})   # time: 0.015791977
    Base.precompile(Tuple{typeof(*),BigFloat,Vector{BigFloat}})   # time: 0.014898929
    Base.precompile(Tuple{typeof(*),BigFloat,Vector{Float64}})   # time: 0.014691288
    Base.precompile(Tuple{typeof(string),String,Type{Nothing},String,Bool,String,Float64,String,Nothing,String,Bool})   # time: 0.007702076
    Base.precompile(Tuple{typeof(string),Int64})   # time: 0.00446557
    Base.precompile(Tuple{typeof(getindex),Tuple,UnitRange{Int64}})   # time: 0.002076376
    Base.precompile(Tuple{typeof(_array_for),Type{Any},UnitRange{Int64},HasShape{1}})   # time: 0.001856792
    Base.precompile(Tuple{typeof(max),BigFloat,Float64})   # time: 0.001537202
    Base.precompile(Tuple{typeof(axes),UnitRange{Int64}})   # time: 0.001144767
    Base.precompile(Tuple{typeof(min),BigFloat,Int64})   # time: 0.00100669
end
