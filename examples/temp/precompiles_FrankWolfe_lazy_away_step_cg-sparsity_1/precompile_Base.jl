function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(getindex),Dict{Int32, Symbol},Int32})   # time: 0.005864829
    Base.precompile(Tuple{typeof(string),Int64})   # time: 0.004750508
    Base.precompile(Tuple{typeof(vect),String,String,String,String,String,String,String})   # time: 0.003563286
    Base.precompile(Tuple{typeof(string),String,Nothing,String,Type{Nothing}})   # time: 0.00346914
    Base.precompile(Tuple{typeof(lstrip),Fix2{typeof(in), Vector{Char}},SubString{String}})   # time: 0.00310529
    Base.precompile(Tuple{typeof(rstrip),Fix2{typeof(in), Vector{Char}},String})   # time: 0.002663432
    Base.precompile(Tuple{typeof(copyto!),Vector{Int64},UnitRange{Int64}})   # time: 0.001745288
end
