function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(getindex),Dict{Int32, Symbol},Int32})   # time: 0.008422027
    Base.precompile(Tuple{typeof(string),Int64})   # time: 0.006611705
    Base.precompile(Tuple{typeof(mod),Int64,Float64})   # time: 0.001396172
    Base.precompile(Tuple{typeof(/),UInt64,Float64})   # time: 0.001273048
end
