function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(getindex),Dict{Int32, Symbol},Int32})   # time: 0.014327171
    Base.precompile(Tuple{typeof(string),Int64})   # time: 0.011796771
    Base.precompile(Tuple{typeof(max),Int64,Float64})   # time: 0.00229091
    Base.precompile(Tuple{typeof(mod),Int64,Float64})   # time: 0.001727297
    Base.precompile(Tuple{typeof(max),Float64,Int64})   # time: 0.00169929
    Base.precompile(Tuple{typeof(mod),Int64,Int64})   # time: 0.001570367
    Base.precompile(Tuple{typeof(/),Float64,Int64})   # time: 0.001334025
    Base.precompile(Tuple{typeof(/),UInt64,Float64})   # time: 0.001257344
    Base.precompile(Tuple{typeof(==),Float64,Int64})   # time: 0.001076244
end
