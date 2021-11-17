function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(mod),Int64,Float64})   # time: 0.043913353
    Base.precompile(Tuple{typeof(-),Vector{Float64},Vector{Float64}})   # time: 0.031937495
    Base.precompile(Tuple{typeof(*),Float64,Vector{Float64}})   # time: 0.022031588
    Base.precompile(Tuple{typeof(getindex),Dict{Int32, Symbol},Int32})   # time: 0.015780259
    Base.precompile(Tuple{typeof(string),Int64})   # time: 0.010548658
    Base.precompile(Tuple{typeof(string),String,Nothing,String,Type{Nothing}})   # time: 0.008146927
    Base.precompile(Tuple{typeof(vect),String,String,String,String,String,String,String})   # time: 0.007554129
    Base.precompile(Tuple{typeof(lstrip),Fix2{typeof(in), Vector{Char}},SubString{String}})   # time: 0.005708392
    Base.precompile(Tuple{typeof(rstrip),Fix2{typeof(in), Vector{Char}},String})   # time: 0.005251145
    Base.precompile(Tuple{typeof(_isdisjoint),Tuple{UInt64},Tuple{UInt64, UInt64}})   # time: 0.002911567
    Base.precompile(Tuple{typeof(max),Float64,Int64})   # time: 0.001950285
    Base.precompile(Tuple{typeof(flush),TTY})   # time: 0.001418173
    Base.precompile(Tuple{typeof(/),UInt64,Float64})   # time: 0.001174086
end
