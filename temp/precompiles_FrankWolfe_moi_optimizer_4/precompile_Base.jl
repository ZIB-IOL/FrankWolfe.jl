function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(getindex),Dict{Int32, Symbol},Int32})   # time: 0.014076878
    Base.precompile(Tuple{typeof(string),Int64})   # time: 0.011022607
    Base.precompile(Tuple{typeof(copyto!),Vector{Int64},UnitRange{Int64}})   # time: 0.004044194
    Base.precompile(Tuple{typeof(mod),Int64,Float64})   # time: 0.003613768
    Base.precompile(Tuple{typeof(max),Float64,Int64})   # time: 0.002292591
    Base.precompile(Tuple{typeof(/),UInt64,Float64})   # time: 0.001330813
end
