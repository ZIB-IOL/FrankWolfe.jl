function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(getindex),Dict{Int32, Symbol},Int32})   # time: 0.005623814
    Base.precompile(Tuple{typeof(string),Int64})   # time: 0.004387064
    Base.precompile(Tuple{typeof(string),String,Nothing,String,Type{Nothing}})   # time: 0.003482488
    Base.precompile(Tuple{typeof(vect),String,String,String,String,String,String,String})   # time: 0.00341558
    Base.precompile(Tuple{typeof(lstrip),Fix2{typeof(in), Vector{Char}},SubString{String}})   # time: 0.002656267
    Base.precompile(Tuple{typeof(rstrip),Fix2{typeof(in), Vector{Char}},String})   # time: 0.002031819
    Base.precompile(Tuple{typeof(copyto!),Vector{Int64},UnitRange{Int64}})   # time: 0.001661029
    Base.precompile(Tuple{typeof(mod),Int64,Float64})   # time: 0.001076531
    Base.precompile(Tuple{typeof(max),Float64,Int64})   # time: 0.001054459
end
