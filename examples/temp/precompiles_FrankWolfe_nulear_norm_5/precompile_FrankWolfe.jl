function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{Core.kwftype(typeof(blended_pairwise_conditional_gradient)),NamedTuple{(:epsilon, :max_iteration, :print_iter, :trajectory, :verbose, :linesearch_tol, :line_search, :emphasis), Tuple{Float64, Int64, Float64, Bool, Bool, Float64, Adaptive, Emphasis}},typeof(blended_pairwise_conditional_gradient),Function,Function,NuclearNormLMO{Float64},RankOneMatrix{Float64, Vector{Float64}, Vector{Float64}}})   # time: 0.7112858
    Base.precompile(Tuple{Core.kwftype(typeof(print_callback)),NamedTuple{(:print_header,), Tuple{Bool}},typeof(print_callback),NTuple{8, String},String})   # time: 0.012743409
    isdefined(FrankWolfe, Symbol("#push_trajectory!#9")) && Base.precompile(Tuple{getfield(FrankWolfe, Symbol("#push_trajectory!#9")),NamedTuple{(:t, :primal, :dual, :dual_gap, :time, :x, :v, :active_set_length, :gamma), _A} where _A<:Tuple{Int64, Float64, Any, Any, Float64, Matrix{Float64}, RankOneMatrix, Int64, Union{Float64, Int64, BigFloat}}})   # time: 0.004735228
end
