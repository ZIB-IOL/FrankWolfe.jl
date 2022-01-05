function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(mul!),Vector{Float64},UniformScaling{Bool},Vector{Float64},Float64,Float64})   # time: 0.08038953
end
