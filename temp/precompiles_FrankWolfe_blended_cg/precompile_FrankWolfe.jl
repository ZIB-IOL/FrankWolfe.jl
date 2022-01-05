function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    isdefined(FrankWolfe, Symbol("#push_trajectory!#33")) && Base.precompile(Tuple{getfield(FrankWolfe, Symbol("#push_trajectory!#33")),NamedTuple{(:t, :primal, :dual, :dual_gap, :time, :x), Tuple{Int64, Float64, Float64, Float64, Float64, Vector{Float64}}}})   # time: 0.08297826
    isdefined(FrankWolfe, Symbol("#reduced_grad!#101")) && Base.precompile(Tuple{getfield(FrankWolfe, Symbol("#reduced_grad!#101")),Vector{Float64},Vector{Float64}})   # time: 0.001148117
end
