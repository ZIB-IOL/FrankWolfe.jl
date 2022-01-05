function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(getindex),Dict{Int32, Symbol},Int32})   # time: 0.013307394
    Base.precompile(Tuple{typeof(string),Int64})   # time: 0.010534548
    Base.precompile(Tuple{typeof(string),String,Type{Nothing},String,Bool,String,Float64})   # time: 0.009778819
    Base.precompile(Tuple{typeof(isapprox),Float64,Float64})   # time: 0.004329018
    Base.precompile(Tuple{typeof(_similar_shape),UnitRange{Int64},HasShape{1}})   # time: 0.00268876
    Base.precompile(Tuple{typeof(max),Int64,Float64})   # time: 0.002597476
    Base.precompile(Tuple{typeof(mod),Int64,Float64})   # time: 0.002132278
    Base.precompile(Tuple{typeof(mod),Int64,Int64})   # time: 0.001513009
    Base.precompile(Tuple{Colon,Int64,Int64})   # time: 0.001428644
    Base.precompile(Tuple{typeof(/),UInt64,Float64})   # time: 0.001365224
    Base.precompile(Tuple{typeof(==),Bool,Int64})   # time: 0.001240913
end
