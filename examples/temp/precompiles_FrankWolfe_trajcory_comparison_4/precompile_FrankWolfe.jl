function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{Core.kwftype(typeof(frank_wolfe)),NamedTuple{(:max_iteration, :line_search, :print_iter, :emphasis, :verbose, :trajectory), Tuple{Int64, Agnostic, Float64, Emphasis, Bool, Bool}},typeof(frank_wolfe),Function,Function,LpNormLMO{Float64, 1},SparseVector{Float64, Int64}})   # time: 0.08101941
    isdefined(FrankWolfe, Symbol("#push_trajectory!#9")) && Base.precompile(Tuple{getfield(FrankWolfe, Symbol("#push_trajectory!#9")),NamedTuple{(:t, :primal, :dual, :dual_gap, :time, :x, :v, :gamma), Tuple{Int64, Float64, Float64, Float64, Float64, SparseVector{Float64, Int64}, ScaledHotVector{Float64}, Rational{Int64}}}})   # time: 0.018419366
end
