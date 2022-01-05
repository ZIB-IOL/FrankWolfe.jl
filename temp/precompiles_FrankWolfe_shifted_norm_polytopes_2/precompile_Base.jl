function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(mod),Int64,Int64})   # time: 0.01710431
    Base.precompile(Tuple{typeof(getindex),Dict{Int32, Symbol},Int32})   # time: 0.005481746
    Base.precompile(Tuple{typeof(string),Int64})   # time: 0.00470015
    Base.precompile(Tuple{typeof(max),Float64,Int64})   # time: 0.001020305
end
