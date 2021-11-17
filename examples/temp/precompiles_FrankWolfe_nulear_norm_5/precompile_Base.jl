function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(getindex),Dict{Int32, Symbol},Int32})   # time: 0.011975217
    Base.precompile(Tuple{typeof(string),Int64})   # time: 0.009167001
    Base.precompile(Tuple{typeof(mod),Int64,Float64})   # time: 0.002110283
    Base.precompile(Tuple{typeof(mod),Int64,Int64})   # time: 0.001338382
    Base.precompile(Tuple{typeof(/),UInt64,Float64})   # time: 0.001214835
end
