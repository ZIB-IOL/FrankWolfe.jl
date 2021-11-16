function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(getindex),Dict{Int32, Symbol},Int32})   # time: 0.031136366
    Base.precompile(Tuple{typeof(string),Int64})   # time: 0.004364847
end
