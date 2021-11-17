function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(min),Float64,Int64})   # time: 0.031887926
    Base.precompile(Tuple{typeof(getindex),Dict{Int32, Symbol},Int32})   # time: 0.010043169
    Base.precompile(Tuple{typeof(string),Int64})   # time: 0.00790707
    Base.precompile(Tuple{typeof(mod),Int64,Float64})   # time: 0.00154454
end
