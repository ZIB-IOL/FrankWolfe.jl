function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(getindex),Dict{Int32, Symbol},Int32})   # time: 0.013485655
    Base.precompile(Tuple{typeof(string),Int64})   # time: 0.010037037
    Base.precompile(Tuple{typeof(string),String,Type{Vector{Float64}},String,Bool,String,Float64,String,Nothing,String,Bool})   # time: 0.007621689
    Base.precompile(Tuple{typeof(getindex),Tuple,UnitRange{Int64}})   # time: 0.002245777
    Base.precompile(Tuple{typeof(max),Int64,Float64})   # time: 0.002139062
    Base.precompile(Tuple{typeof(mod),Int64,Int64})   # time: 0.001899432
    Base.precompile(Tuple{typeof(max),Float64,Int64})   # time: 0.001580183
    Base.precompile(Tuple{typeof(_similar_shape),UnitRange{Int64},HasShape{1}})   # time: 0.001448112
    Base.precompile(Tuple{typeof(/),UInt64,Float64})   # time: 0.001311428
    Base.precompile(Tuple{typeof(flush),TTY})   # time: 0.001261633
    Base.precompile(Tuple{Colon,Int64,Int64})   # time: 0.001193048
    Base.precompile(Tuple{typeof(/),Float64,Int64})   # time: 0.001149538
end
