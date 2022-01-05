function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(getindex),Dict{Int32, Symbol},Int32})   # time: 0.055868015
    Base.precompile(Tuple{typeof(string),Int64})   # time: 0.010229584
    Base.precompile(Tuple{typeof(mod),Int64,Float64})   # time: 0.00322741
    Base.precompile(Tuple{typeof(max),Float64,Int64})   # time: 0.002396378
    Base.precompile(Tuple{typeof(/),Float64,Int64})   # time: 0.001309656
    Base.precompile(Tuple{typeof(/),UInt64,Float64})   # time: 0.001293034
end
