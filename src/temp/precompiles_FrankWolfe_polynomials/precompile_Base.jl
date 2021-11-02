function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(string),Int64})   # time: 0.07094364
    Base.precompile(Tuple{typeof(getindex),Dict{Int32, Symbol},Int32})   # time: 0.017504528
    Base.precompile(Tuple{typeof(string),String,Type{Vector{Float64}},String,Bool,String,Float64,String,Nothing,String,Bool})   # time: 0.009162244
    Base.precompile(Tuple{typeof(getindex),Tuple,UnitRange{Int64}})   # time: 0.005158042
    Base.precompile(Tuple{typeof(_array_for),Type{Any},UnitRange{Int64},HasShape{1}})   # time: 0.003424937
    Base.precompile(Tuple{typeof(axes),UnitRange{Int64}})   # time: 0.002884434
    Base.precompile(Tuple{typeof(max),Int64,Float64})   # time: 0.002456729
    Base.precompile(Tuple{typeof(min),Float64,Int64})   # time: 0.002113532
    Base.precompile(Tuple{typeof(mod),Int64,Int64})   # time: 0.001708451
    Base.precompile(Tuple{typeof(/),Float64,Int64})   # time: 0.001409532
    Base.precompile(Tuple{typeof(/),UInt64,Float64})   # time: 0.001407713
end
