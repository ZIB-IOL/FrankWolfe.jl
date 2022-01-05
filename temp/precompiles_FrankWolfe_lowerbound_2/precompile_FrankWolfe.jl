function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{Core.kwftype(typeof(away_frank_wolfe)),NamedTuple{(:max_iteration, :line_search, :print_iter, :memory_mode, :verbose, :epsilon, :trajectory), Tuple{Int64, Adaptive{Float64, Int64}, Float64, InplaceEmphasis, Bool, Float64, Bool}},typeof(away_frank_wolfe),Function,Function,ProbabilitySimplexOracle{Float64},ScaledHotVector{Float64}})   # time: 0.746063
    isdefined(FrankWolfe, Symbol("#push_trajectory!#33")) && Base.precompile(Tuple{getfield(FrankWolfe, Symbol("#push_trajectory!#33")),NamedTuple{(:t, :primal, :dual, :dual_gap, :time, :x, :v, :active_set_length, :gamma), Tuple{Int64, Float64, Float64, Float64, Float64, Vector{Float64}, ScaledHotVector{Float64}, Int64, Float64}}})   # time: 0.034999933
    Base.precompile(Tuple{Core.kwftype(typeof(print_callback)),NamedTuple{(:print_header,), Tuple{Bool}},typeof(print_callback),NTuple{8, String},String})   # time: 0.014442689
    Base.precompile(Tuple{Type{Adaptive}})   # time: 0.004269498
end
