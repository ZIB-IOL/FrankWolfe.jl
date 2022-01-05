function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(getindex),Dict{Int32, Symbol},Int32})   # time: 0.013314502
    Base.precompile(Tuple{typeof(string),Int64})   # time: 0.010884685
    Base.precompile(Tuple{typeof(mod),Int64,Float64})   # time: 0.003203873
    Base.precompile(Tuple{typeof(max),Float64,Int64})   # time: 0.002334007
    Base.precompile(Tuple{typeof(/),UInt64,Float64})   # time: 0.001655572
    Base.precompile(Tuple{typeof(/),Float64,Int64})   # time: 0.001201749
end
