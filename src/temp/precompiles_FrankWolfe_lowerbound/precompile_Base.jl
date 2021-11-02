function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(getindex),Dict{Int32, Symbol},Int32})   # time: 0.014370585
    Base.precompile(Tuple{typeof(string),Int64})   # time: 0.011022037
    Base.precompile(Tuple{typeof(mod),Int64,Float64})   # time: 0.002679033
    Base.precompile(Tuple{typeof(/),UInt64,Float64})   # time: 0.001489083
end
