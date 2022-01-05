function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(dot),Vector{Float64},Vector{Float64}})   # time: 0.007324696
end
