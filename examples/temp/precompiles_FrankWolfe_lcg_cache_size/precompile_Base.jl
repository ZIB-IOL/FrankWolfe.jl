function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(vect),String,String,String,String,String,String,String,String})   # time: 0.034779057
    Base.precompile(Tuple{typeof(string),String,Type{Nothing},String,Float64,String,Bool})   # time: 0.007212898
    Base.precompile(Tuple{typeof(getindex),Dict{Int32, Symbol},Int32})   # time: 0.005901991
    Base.precompile(Tuple{typeof(string),Int64})   # time: 0.004660441
    Base.precompile(Tuple{typeof(zero),Type{BigFloat}})   # time: 0.001996452
    Base.precompile(Tuple{typeof(copyto!),Vector{Int64},UnitRange{Int64}})   # time: 0.001716391
    Base.precompile(Tuple{typeof(max),BigFloat,Float64})   # time: 0.001603808
end
