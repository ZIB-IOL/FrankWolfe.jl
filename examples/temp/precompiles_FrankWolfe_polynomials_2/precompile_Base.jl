function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(string),String,BigFloat})   # time: 0.0872468
    Base.precompile(Tuple{typeof(getindex),Dict{Int32, Symbol},Int32})   # time: 0.009819023
    Base.precompile(Tuple{typeof(maximum),Vector{Float64}})   # time: 0.009663084
    Base.precompile(Tuple{typeof(minimum),Vector{Float64}})   # time: 0.007248462
    Base.precompile(Tuple{typeof(sum),Vector{Float64}})   # time: 0.004953483
    Base.precompile(Tuple{typeof(string),String,Float64,String,Int64,String})   # time: 0.003360272
    Base.precompile(Tuple{typeof(_array_for),Type{Any},UnitRange{Int64},HasShape{1}})   # time: 0.003306698
    Base.precompile(Tuple{typeof(getindex),Tuple,UnitRange{Int64}})   # time: 0.002880039
    Base.precompile(Tuple{typeof(axes),UnitRange{Int64}})   # time: 0.001846596
    Base.precompile(Tuple{typeof(max),Int64,BigFloat})   # time: 0.001743373
    Base.precompile(Tuple{typeof(/),Float64,Int64})   # time: 0.001030923
    Base.precompile(Tuple{typeof(/),UInt64,Float64})   # time: 0.0010303
end
