function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(getindex),Dict{Int32, Symbol},Int32})   # time: 0.005426368
    Base.precompile(Tuple{typeof(string),Int64})   # time: 0.004377697
    Base.precompile(Tuple{typeof(//),Int64,Int64})   # time: 0.002081053
end
