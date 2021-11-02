function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(getindex),Dict{Int32, Symbol},Int32})   # time: 0.015506861
    Base.precompile(Tuple{typeof(string),Int64})   # time: 0.011352655
    Base.precompile(Tuple{typeof(mod),Int64,Float64})   # time: 0.00246085
    Base.precompile(Tuple{typeof(max),Float64,Int64})   # time: 0.002144748
    Base.precompile(Tuple{typeof(/),UInt64,Float64})   # time: 0.001551866
end
