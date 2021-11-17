function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(getindex),Dict{Int32, Symbol},Int32})   # time: 0.016574478
    Base.precompile(Tuple{typeof(string),Int64})   # time: 0.011869763
    Base.precompile(Tuple{typeof(mod),Int64,Float64})   # time: 0.002755892
    Base.precompile(Tuple{typeof(max),Float64,Int64})   # time: 0.002011886
    Base.precompile(Tuple{typeof(/),UInt64,Float64})   # time: 0.001256913
end
