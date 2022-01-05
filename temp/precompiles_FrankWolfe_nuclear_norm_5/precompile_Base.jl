function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(string),Int64})   # time: 0.049967814
    Base.precompile(Tuple{typeof(getindex),Dict{Int32, Symbol},Int32})   # time: 0.013162557
    Base.precompile(Tuple{typeof(max),Int64,Float64})   # time: 0.002381289
    Base.precompile(Tuple{typeof(mod),Int64,Float64})   # time: 0.00233324
    Base.precompile(Tuple{typeof(mod),Int64,Int64})   # time: 0.001446823
    Base.precompile(Tuple{typeof(/),UInt64,Float64})   # time: 0.001189806
    Base.precompile(Tuple{typeof(==),Bool,Int64})   # time: 0.001056623
end
