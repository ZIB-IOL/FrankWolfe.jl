function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{Core.kwftype(typeof(frank_wolfe)),NamedTuple{(:max_iteration, :line_search, :print_iter, :memory_mode, :verbose, :trajectory, :momentum), Tuple{Int64, Shortstep{Float64}, Float64, OutplaceEmphasis, Bool, Bool, Float64}},typeof(frank_wolfe),Function,Function,LpNormLMO{Float64, 1},SparseVector{Float64, Int64}})   # time: 0.10808421
    isdefined(FrankWolfe, Symbol("#push_trajectory!#33")) && Base.precompile(Tuple{getfield(FrankWolfe, Symbol("#push_trajectory!#33")),NamedTuple{(:t, :primal, :dual, :dual_gap, :time, :x, :v, :gamma), Tuple{Int64, Float64, Float64, Float64, Float64, SparseVector{Float64, Int64}, ScaledHotVector{Float64}, Float64}}})   # time: 0.002747061
    Base.precompile(Tuple{Core.kwftype(typeof(print_callback)),NamedTuple{(:print_header,), Tuple{Bool}},typeof(print_callback),Vector{String},String})   # time: 0.001049784
end
