function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{Core.kwftype(typeof(frank_wolfe)),NamedTuple{(:epsilon, :max_iteration, :print_iter, :trajectory, :verbose, :linesearch_tol, :line_search, :emphasis, :gradient), Tuple{Float64, Int64, Float64, Bool, Bool, Float64, Adaptive, Emphasis, SparseArrays.SparseMatrixCSC{Float64, Int64}}},typeof(frank_wolfe),Function,Function,NuclearNormLMO{Float64},RankOneMatrix{Float64, Vector{Float64}, Vector{Float64}}})   # time: 2.2658193
    Base.precompile(Tuple{typeof(print_callback),Tuple{String, String, Float64, Any, Any, Float64, Float64},String})   # time: 0.11209728
    Base.precompile(Tuple{typeof(fast_dot),SparseArrays.SparseMatrixCSC{BigFloat, Int64},Matrix{T} where T})   # time: 0.01771594
    isdefined(FrankWolfe, Symbol("#push_trajectory!#9")) && Base.precompile(Tuple{getfield(FrankWolfe, Symbol("#push_trajectory!#9")),NamedTuple{(:t, :primal, :dual, :dual_gap, :time, :x, :v, :gamma), Tuple{Int64, Float64, Float64, Float64, Float64, Matrix{Float64}, RankOneMatrix{Float64, Vector{Float64}, Vector{Float64}}, Float64}}})   # time: 0.012130197
    isdefined(FrankWolfe, Symbol("#push_trajectory!#9")) && Base.precompile(Tuple{getfield(FrankWolfe, Symbol("#push_trajectory!#9")),NamedTuple{(:t, :primal, :dual, :dual_gap, :time, :x, :v, :gamma), _A} where _A<:Tuple{Int64, Float64, Any, Any, Float64, Union{RankOneMatrix{Float64, Vector{Float64}, Vector{Float64}}, Matrix{Float64}}, RankOneMatrix, Any}})   # time: 0.004179441
    Base.precompile(Tuple{Core.kwftype(typeof(print_callback)),NamedTuple{(:print_header,), Tuple{Bool}},typeof(print_callback),Vector{String},String})   # time: 0.0021596
end
