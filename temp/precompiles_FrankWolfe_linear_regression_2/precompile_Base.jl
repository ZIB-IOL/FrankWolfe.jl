function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(getindex),Dict{Int32, Symbol},Int32})   # time: 0.015037166
    Base.precompile(Tuple{typeof(string),Int64})   # time: 0.012419875
    Base.precompile(Tuple{typeof(mod),Int64,Float64})   # time: 0.004315312
    Base.precompile(Tuple{typeof(/),UInt64,Float64})   # time: 0.001627842
end
