function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(getindex),Dict{Int32, Symbol},Int32})   # time: 0.03787193
    Base.precompile(Tuple{typeof(string),Int64})   # time: 0.01197674
    Base.precompile(Tuple{typeof(string),String,Type{Nothing},String,Bool,String,Float64,String,Nothing,String,Bool})   # time: 0.007227053
    Base.precompile(Tuple{typeof(max),Int64,Float64})   # time: 0.002882442
    Base.precompile(Tuple{typeof(getindex),Tuple,UnitRange{Int64}})   # time: 0.002712663
    Base.precompile(Tuple{typeof(mod),Int64,Float64})   # time: 0.002522972
    Base.precompile(Tuple{typeof(/),Float64,Int64})   # time: 0.002032696
    Base.precompile(Tuple{typeof(mod),Int64,Int64})   # time: 0.001777364
    Base.precompile(Tuple{typeof(max),Float64,Int64})   # time: 0.001762977
    Base.precompile(Tuple{typeof(_similar_shape),UnitRange{Int64},HasShape{1}})   # time: 0.001613023
    Base.precompile(Tuple{typeof(/),UInt64,Float64})   # time: 0.001332273
    Base.precompile(Tuple{Colon,Int64,Int64})   # time: 0.001281913
end
