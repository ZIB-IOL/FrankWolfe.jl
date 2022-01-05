function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(getindex),Dict{Int32, Symbol},Int32})   # time: 0.005922431
    Base.precompile(Tuple{typeof(string),Int64})   # time: 0.004536226
    Base.precompile(Tuple{typeof(mod),Int64,Float64})   # time: 0.001393322
end
