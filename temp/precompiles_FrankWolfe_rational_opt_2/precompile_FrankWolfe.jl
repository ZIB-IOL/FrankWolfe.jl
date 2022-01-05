function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{Core.kwftype(typeof(frank_wolfe)),NamedTuple{(:max_iteration, :line_search, :print_iter, :verbose, :memory_mode), Tuple{Int64, Shortstep{Rational{Int64}}, Float64, Bool, OutplaceEmphasis}},typeof(frank_wolfe),Function,Function,ProbabilitySimplexOracle{Rational{BigInt}},ScaledHotVector{Rational{BigInt}}})   # time: 0.141858
    isdefined(FrankWolfe, Symbol("#push_trajectory!#33")) && Base.precompile(Tuple{getfield(FrankWolfe, Symbol("#push_trajectory!#33")),NamedTuple{(:t, :primal, :dual, :dual_gap, :time, :x, :v, :gamma), Tuple{Int64, Rational{BigInt}, Rational{BigInt}, Rational{BigInt}, Float64, ScaledHotVector{Rational{BigInt}}, ScaledHotVector{Rational{BigInt}}, BigFloat}}})   # time: 0.04731252
    Base.precompile(Tuple{Type{Shortstep},Rational{Int64}})   # time: 0.00955542
end
