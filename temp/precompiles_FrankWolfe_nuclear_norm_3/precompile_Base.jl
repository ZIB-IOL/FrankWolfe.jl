function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(getindex),Dict{Int32, Symbol},Int32})   # time: 0.015452829
    Base.precompile(Tuple{typeof(string),Int64})   # time: 0.011498383
    Base.precompile(Tuple{typeof(max),Int64,Float64})   # time: 0.002477537
    Base.precompile(Tuple{typeof(mod),Int64,Int64})   # time: 0.002170096
    Base.precompile(Tuple{typeof(mod),Int64,Float64})   # time: 0.002080139
    Base.precompile(Tuple{typeof(max),Float64,Int64})   # time: 0.002051504
    Base.precompile(Tuple{typeof(/),Float64,Int64})   # time: 0.001465472
    Base.precompile(Tuple{typeof(/),UInt64,Float64})   # time: 0.001183383
end
