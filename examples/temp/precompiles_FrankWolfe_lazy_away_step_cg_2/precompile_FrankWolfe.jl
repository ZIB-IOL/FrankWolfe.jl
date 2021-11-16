function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{Core.kwftype(typeof(away_frank_wolfe)),NamedTuple{(:max_iteration, :line_search, :print_iter, :epsilon, :emphasis, :verbose, :away_steps, :trajectory), Tuple{Int64, Adaptive, Float64, Float64, Emphasis, Bool, Bool, Bool}},typeof(away_frank_wolfe),Function,Function,KSparseLMO{Float64},SparseVector{Float64, Int64}})   # time: 0.003852925
end
