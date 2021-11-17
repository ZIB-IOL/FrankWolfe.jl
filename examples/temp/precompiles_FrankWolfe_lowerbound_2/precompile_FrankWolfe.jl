function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{Core.kwftype(typeof(away_frank_wolfe)),NamedTuple{(:max_iteration, :line_search, :print_iter, :emphasis, :verbose, :epsilon, :trajectory), Tuple{Int64, Adaptive, Float64, Emphasis, Bool, Float64, Bool}},typeof(away_frank_wolfe),Function,Function,ProbabilitySimplexOracle{Float64},ScaledHotVector{Float64}})   # time: 0.2808833
    isdefined(FrankWolfe, Symbol("#push_trajectory!#9")) && Base.precompile(Tuple{getfield(FrankWolfe, Symbol("#push_trajectory!#9")),NamedTuple{(:t, :primal, :dual, :dual_gap, :time, :x, :v, :active_set_length, :gamma), Tuple{Int64, Float64, Float64, Float64, Float64, SparseVector{Float64, Int64}, ScaledHotVector{Float64}, Int64, Float64}}})   # time: 0.012031294
    Base.precompile(Tuple{Core.kwftype(typeof(print_callback)),NamedTuple{(:print_header,), Tuple{Bool}},typeof(print_callback),NTuple{8, String},String})   # time: 0.010515343
    Base.precompile(Tuple{typeof(print_callback),Tuple{String, String, Float64, Float64, Float64, Float64, Float64, Int64},String})   # time: 0.007608741
end
