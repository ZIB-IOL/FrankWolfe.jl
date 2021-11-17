function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(*),Union{Float16, Float32, Float64, Signed},DoubleFloats.DoubleFloat{T<:Union{Core.Float16, Core.Float32, Core.Float64}},Float64})   # time: 0.022145651
end
