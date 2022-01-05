function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(getindex),Dict{Int32, Symbol},Int32})   # time: 0.005513862
    Base.precompile(Tuple{typeof(string),Int64})   # time: 0.004305556
    Base.precompile(Tuple{typeof(copyto!),Vector{Int64},UnitRange{Int64}})   # time: 0.001529798
    Base.precompile(Tuple{typeof(mod),Int64,Float64})   # time: 0.001253014
end
