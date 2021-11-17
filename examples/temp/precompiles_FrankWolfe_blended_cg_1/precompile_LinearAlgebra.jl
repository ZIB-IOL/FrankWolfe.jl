function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(*),Transpose{Float64, Vector{Float64}},Matrix{Float64}})   # time: 0.027187847
    Base.precompile(Tuple{typeof(eigmin),Matrix{Float64}})   # time: 0.007285933
    isdefined(LinearAlgebra, Symbol("#11#12")) && Base.precompile(Tuple{getfield(LinearAlgebra, Symbol("#11#12")),Float64,Float64})   # time: 0.005142779
    Base.precompile(Tuple{typeof(dot),Vector{Float64},Matrix{Float64},Vector{Float64}})   # time: 0.002845019
end
