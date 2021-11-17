function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(getindex),Dict{Int32, Symbol},Int32})   # time: 0.014446137
    Base.precompile(Tuple{typeof(string),Int64})   # time: 0.010799942
    Base.precompile(Tuple{typeof(mod),Int64,Float64})   # time: 0.002152222
    Base.precompile(Tuple{typeof(/),UInt64,Float64})   # time: 0.001484264
    Base.precompile(Tuple{typeof(/),Float64,Int64})   # time: 0.00117298
end
