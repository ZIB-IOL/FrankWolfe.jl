function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(hashindex),Tuple{Any},Int64})   # time: 0.007559432
    Base.precompile(Tuple{typeof(hashindex),Tuple{Any, Any},Int64})   # time: 0.007482958
end
