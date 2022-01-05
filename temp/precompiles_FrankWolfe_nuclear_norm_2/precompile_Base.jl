function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(getindex),Dict{Int32, Symbol},Int32})   # time: 0.01427665
    Base.precompile(Tuple{typeof(string),Int64})   # time: 0.010014819
    Base.precompile(Tuple{typeof(mod),Int64,Float64})   # time: 0.003490085
    Base.precompile(Tuple{typeof(/),UInt64,Float64})   # time: 0.001817258
    Base.precompile(Tuple{typeof(/),Float64,Int64})   # time: 0.001154955
end
