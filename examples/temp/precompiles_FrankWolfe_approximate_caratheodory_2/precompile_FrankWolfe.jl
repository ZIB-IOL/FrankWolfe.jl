function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{Core.kwftype(typeof(frank_wolfe)),NamedTuple{(:max_iteration, :line_search, :L, :print_iter, :verbose, :emphasis), Tuple{Int64, RationalShortstep, Int64, Float64, Bool, Emphasis}},typeof(frank_wolfe),Function,Function,ProbabilitySimplexOracle{Rational{BigInt}},ScaledHotVector{Rational{BigInt}}})   # time: 0.21134385
    Base.precompile(Tuple{typeof(print),IOBuffer,RationalShortstep})   # time: 0.008633916
    isdefined(FrankWolfe, Symbol("#push_trajectory!#9")) && Base.precompile(Tuple{getfield(FrankWolfe, Symbol("#push_trajectory!#9")),NamedTuple{(:t, :primal, :dual, :dual_gap, :time, :x, :v, :gamma), _A} where _A<:Tuple{Int64, Union{Float64, Rational{BigInt}}, Union{Float64, Rational{BigInt}, BigFloat}, Union{Float64, Rational{BigInt}}, Float64, Union{ScaledHotVector{Rational{BigInt}}, SparseVector{Rational{BigInt}, Int64}}, ScaledHotVector{Rational{BigInt}}, Rational{BigInt}}})   # time: 0.002246791
end
