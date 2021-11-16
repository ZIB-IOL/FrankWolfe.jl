function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(getindex),Dict{Int32, Symbol},Int32})   # time: 0.005384615
    Base.precompile(Tuple{typeof(string),Int64})   # time: 0.004675794
    Base.precompile(Tuple{typeof(vect),String,String,String,String,String,String,String,String})   # time: 0.003488841
    Base.precompile(Tuple{typeof(_array_for),Type{Any},UnitRange{Int64},HasShape{1}})   # time: 0.002059126
    Base.precompile(Tuple{typeof(getindex),Tuple,UnitRange{Int64}})   # time: 0.001740369
    Base.precompile(Tuple{typeof(mod),Int64,Float64})   # time: 0.001199523
end
