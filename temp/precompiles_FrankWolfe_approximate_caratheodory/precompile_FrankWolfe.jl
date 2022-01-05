function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{Core.kwftype(typeof(frank_wolfe)),NamedTuple{(:max_iteration, :line_search, :print_iter, :verbose, :memory_mode), Tuple{Int64, Agnostic{Float64}, Float64, Bool, OutplaceEmphasis}},typeof(frank_wolfe),Function,Function,ProbabilitySimplexOracle{Rational{BigInt}},ScaledHotVector{Rational{BigInt}}})   # time: 0.2661169
    Base.precompile(Tuple{typeof(print_callback),Tuple{String, String, Float64, Any, Float64, Float64, Float64},String})   # time: 0.004444123
    isdefined(FrankWolfe, Symbol("#push_trajectory!#33")) && Base.precompile(Tuple{getfield(FrankWolfe, Symbol("#push_trajectory!#33")),NamedTuple{(:t, :primal, :dual, :dual_gap, :time, :x, :v, :gamma), _A} where _A<:Tuple{Int64, Union{Float64, Rational{BigInt}, BigFloat}, Any, Union{Float64, Rational{BigInt}, BigFloat}, Float64, Union{ScaledHotVector{Rational{BigInt}}, SparseVector{BigFloat, Int64}}, ScaledHotVector{Rational{BigInt}}, Float64}})   # time: 0.0016796
    Base.precompile(Tuple{typeof(dot),ScaledHotVector{Rational{BigInt}},ScaledHotVector{Rational{BigInt}}})   # time: 0.001534546
    Base.precompile(Tuple{Core.kwftype(typeof(print_callback)),NamedTuple{(:print_header,), Tuple{Bool}},typeof(print_callback),Vector{String},String})   # time: 0.001028549
end
