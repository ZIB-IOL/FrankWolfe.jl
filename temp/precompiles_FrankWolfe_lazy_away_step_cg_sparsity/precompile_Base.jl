function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(getindex),Dict{Int32, Symbol},Int32})   # time: 0.009278669
    Base.precompile(Tuple{typeof(string),Int64})   # time: 0.007095609
    Base.precompile(Tuple{typeof(mod),Int64,Float64})   # time: 0.003884155
    Base.precompile(Tuple{typeof(max),Float64,Int64})   # time: 0.002186344
    Base.precompile(Tuple{typeof(/),UInt64,Float64})   # time: 0.001866055
    Base.precompile(Tuple{typeof(flush),TTY})   # time: 0.001384229
end
