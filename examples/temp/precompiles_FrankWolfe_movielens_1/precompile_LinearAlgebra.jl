function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(norm),Matrix{BigFloat}})   # time: 0.017680502
end
