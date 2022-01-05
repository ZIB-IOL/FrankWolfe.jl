function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{Core.kwftype(typeof(blended_pairwise_conditional_gradient)),NamedTuple{(:epsilon, :max_iteration, :print_iter, :trajectory, :verbose, :line_search, :memory_mode), Tuple{Float64, Int64, Float64, Bool, Bool, Adaptive{Float64, Int64}, InplaceEmphasis}},typeof(blended_pairwise_conditional_gradient),Function,Function,NuclearNormLMO{Float64},RankOneMatrix{Float64, Vector{Float64}, Vector{Float64}}})   # time: 0.34339795
    Base.precompile(Tuple{Core.kwftype(typeof(print_callback)),NamedTuple{(:print_header,), Tuple{Bool}},typeof(print_callback),NTuple{8, String},String})   # time: 0.012937946
    isdefined(FrankWolfe, Symbol("#push_trajectory!#33")) && Base.precompile(Tuple{getfield(FrankWolfe, Symbol("#push_trajectory!#33")),NamedTuple{(:t, :primal, :dual, :dual_gap, :time, :x, :v, :active_set_length, :gamma), Tuple{Int64, Float64, Float64, Float64, Float64, Matrix{Float64}, RankOneMatrix{Float64, Vector{Float64}, Vector{Float64}}, Int64, Float64}}})   # time: 0.005380466
end
