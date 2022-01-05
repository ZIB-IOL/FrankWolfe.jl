function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{Core.kwftype(typeof(frank_wolfe)),NamedTuple{(:max_iteration, :line_search, :print_iter, :memory_mode, :verbose, :trajectory), Tuple{Int64, Shortstep{Float64}, Float64, InplaceEmphasis, Bool, Bool}},typeof(frank_wolfe),Function,Function,ProbabilitySimplexOracle{Float64},Vector{Float64}})   # time: 0.19531982
    Base.precompile(Tuple{Type{Shortstep},Float64})   # time: 0.02998196
    isdefined(FrankWolfe, Symbol("#push_trajectory!#33")) && Base.precompile(Tuple{getfield(FrankWolfe, Symbol("#push_trajectory!#33")),NamedTuple{(:t, :primal, :dual, :dual_gap, :time, :x, :v, :gamma), Tuple{Int64, Float64, Float64, Float64, Float64, Vector{Float64}, ScaledHotVector{Float64}, Float64}}})   # time: 0.017530503
end
