function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(sum),Vector{BigFloat}})   # time: 0.001072475
end
