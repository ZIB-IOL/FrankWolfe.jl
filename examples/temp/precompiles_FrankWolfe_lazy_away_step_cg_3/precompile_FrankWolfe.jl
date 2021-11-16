function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{Core.kwftype(typeof(away_frank_wolfe)),NamedTuple{(:max_iteration, :line_search, :print_iter, :emphasis, :verbose, :epsilon, :trajectory, :away_steps), Tuple{Int64, Adaptive, Float64, Emphasis, Bool, Float64, Bool, Bool}},typeof(away_frank_wolfe),Function,Function,KSparseLMO{Float64},SparseVector{Float64, Int64}})   # time: 0.003758145
end
