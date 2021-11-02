function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(dot),Vector{Float64},Matrix{Float64},Vector{Float64}})   # time: 0.002662907
    Base.precompile(Tuple{typeof(*),Transpose{Float64, Vector{Float64}},Matrix{Float64}})   # time: 0.002119956
end
