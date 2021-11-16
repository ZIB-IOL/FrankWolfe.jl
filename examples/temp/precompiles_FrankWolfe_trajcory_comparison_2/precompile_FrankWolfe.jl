function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{Core.kwftype(typeof(frank_wolfe)),NamedTuple{(:max_iteration, :line_search, :L, :print_iter, :emphasis, :verbose, :trajectory, :momentum), Tuple{Int64, Shortstep, Int64, Float64, Emphasis, Bool, Bool, Float64}},typeof(frank_wolfe),Function,Function,LpNormLMO{Float64, 1},SparseVector{Float64, Int64}})   # time: 0.13679837
    Base.precompile(Tuple{Core.kwftype(typeof(print_callback)),NamedTuple{(:print_header,), Tuple{Bool}},typeof(print_callback),Vector{String},String})   # time: 0.001014022
end
