function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{Core.kwftype(typeof(away_frank_wolfe)),NamedTuple{(:max_iteration, :line_search, :L, :print_iter, :epsilon, :momentum, :emphasis, :verbose, :away_steps, :trajectory), Tuple{Int64, Adaptive, Int64, Float64, Float64, Float64, Emphasis, Bool, Bool, Bool}},typeof(away_frank_wolfe),Function,Function,KSparseLMO{Float64},SparseVector{Float64, Int64}})   # time: 0.10256607
    Base.precompile(Tuple{Core.kwftype(typeof(print_callback)),NamedTuple{(:print_header,), Tuple{Bool}},typeof(print_callback),NTuple{8, String},String})   # time: 0.003748997
end
