function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(getindex),Dict{Int32, Symbol},Int32})   # time: 0.010375538
    Base.precompile(Tuple{typeof(string),Int64})   # time: 0.007643581
    Base.precompile(Tuple{typeof(mod),Int64,Float64})   # time: 0.004029379
    Base.precompile(Tuple{typeof(/),UInt64,Float64})   # time: 0.002092528
    Base.precompile(Tuple{typeof(max),Float64,Int64})   # time: 0.001429752
    Base.precompile(Tuple{typeof(flush),TTY})   # time: 0.001289164
    Base.precompile(Tuple{typeof(iterate),UnitRange{Int64}})   # time: 0.001232664
end
