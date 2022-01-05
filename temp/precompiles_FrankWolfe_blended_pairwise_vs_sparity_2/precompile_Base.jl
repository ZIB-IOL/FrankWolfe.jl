function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(getindex),Dict{Int32, Symbol},Int32})   # time: 0.017015489
    Base.precompile(Tuple{typeof(string),Int64})   # time: 0.013640775
    Base.precompile(Tuple{typeof(string),String,Type{Nothing},String,Bool,String,Float64,String,Nothing,String,Bool})   # time: 0.008609676
    Base.precompile(Tuple{typeof(getindex),Tuple,UnitRange{Int64}})   # time: 0.003154719
    Base.precompile(Tuple{typeof(max),Int64,Float64})   # time: 0.002287242
    Base.precompile(Tuple{typeof(mod),Int64,Int64})   # time: 0.001944822
    Base.precompile(Tuple{typeof(mod),Int64,Float64})   # time: 0.001907389
    Base.precompile(Tuple{typeof(_similar_shape),UnitRange{Int64},HasShape{1}})   # time: 0.001661339
    Base.precompile(Tuple{typeof(max),Float64,Int64})   # time: 0.001624189
    Base.precompile(Tuple{typeof(/),UInt64,Float64})   # time: 0.001273873
    Base.precompile(Tuple{Colon,Int64,Int64})   # time: 0.001266329
end
