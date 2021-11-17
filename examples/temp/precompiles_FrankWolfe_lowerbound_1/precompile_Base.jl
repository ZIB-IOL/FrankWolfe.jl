function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(getindex),Dict{Int32, Symbol},Int32})   # time: 0.015418617
    Base.precompile(Tuple{typeof(string),Int64})   # time: 0.010832351
    Base.precompile(Tuple{typeof(string),String,Nothing,String,Type{Nothing}})   # time: 0.008031974
    Base.precompile(Tuple{typeof(vect),String,String,String,String,String,String,String})   # time: 0.00795857
    Base.precompile(Tuple{typeof(lstrip),Fix2{typeof(in), Vector{Char}},SubString{String}})   # time: 0.007181251
    Base.precompile(Tuple{typeof(rstrip),Fix2{typeof(in), Vector{Char}},String})   # time: 0.004481436
    Base.precompile(Tuple{typeof(copyto!),Vector{Int64},UnitRange{Int64}})   # time: 0.003980962
    Base.precompile(Tuple{typeof(//),Int64,Int64})   # time: 0.002612792
    Base.precompile(Tuple{typeof(_isdisjoint),Tuple{UInt64, UInt64},Tuple{UInt64}})   # time: 0.002430298
    Base.precompile(Tuple{typeof(mod),Int64,Float64})   # time: 0.002089549
    Base.precompile(Tuple{typeof(flush),TTY})   # time: 0.00143249
    Base.precompile(Tuple{typeof(/),UInt64,Float64})   # time: 0.001383181
    Base.precompile(Tuple{typeof(_isdisjoint),Tuple{UInt64, UInt64},Tuple{UInt64, UInt64}})   # time: 0.001290925
end
