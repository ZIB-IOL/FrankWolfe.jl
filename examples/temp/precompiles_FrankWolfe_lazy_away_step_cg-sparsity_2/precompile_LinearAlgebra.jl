function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(norm),Vector{BigFloat}})   # time: 0.045040738
end
