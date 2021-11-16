function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(getindex),Dict{Int32, Symbol},Int32})   # time: 0.005390258
    Base.precompile(Tuple{typeof(string),Int64})   # time: 0.004138913
end
