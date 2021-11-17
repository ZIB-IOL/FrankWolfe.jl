function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(>=),Rational{BigInt},Float64})   # time: 0.018024068
    Base.precompile(Tuple{typeof(-),Vector{Rational{BigInt}}})   # time: 0.016684713
    Base.precompile(Tuple{typeof(getindex),Dict{Int32, Symbol},Int32})   # time: 0.006551353
    Base.precompile(Tuple{typeof(string),Int64})   # time: 0.004671033
    Base.precompile(Tuple{typeof(string),String,Nothing,String,Type{Nothing}})   # time: 0.004377735
    Base.precompile(Tuple{typeof(vect),String,String,String,String,String,String,String})   # time: 0.003858832
    Base.precompile(Tuple{typeof(lstrip),Fix2{typeof(in), Vector{Char}},SubString{String}})   # time: 0.002610913
    Base.precompile(Tuple{typeof(copyto!),Vector{Int64},UnitRange{Int64}})   # time: 0.001873559
    Base.precompile(Tuple{typeof(rstrip),Fix2{typeof(in), Vector{Char}},String})   # time: 0.001857905
    Base.precompile(Tuple{typeof(_isdisjoint),Tuple{UInt64, UInt64},Tuple{UInt64}})   # time: 0.001604054
    Base.precompile(Tuple{typeof(isless),Rational{BigInt},Rational{BigInt}})   # time: 0.001094771
    Base.precompile(Tuple{typeof(//),Int64,Int64})   # time: 0.001087238
end
