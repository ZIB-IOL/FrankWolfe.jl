function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{Core.kwftype(typeof(lazified_conditional_gradient)),NamedTuple{(:epsilon, :max_iteration, :print_iter, :trajectory, :verbose, :linesearch_tol, :line_search, :emphasis, :gradient), Tuple{Float64, Int64, Float64, Bool, Bool, Float64, Adaptive, Emphasis, SparseArrays.SparseMatrixCSC{Float64, Int64}}},typeof(lazified_conditional_gradient),Function,Function,NuclearNormLMO{Float64},RankOneMatrix{Float64, Vector{Float64}, Vector{Float64}}})   # time: 1.1163838
    isdefined(FrankWolfe, Symbol("#push_trajectory!#9")) && Base.precompile(Tuple{getfield(FrankWolfe, Symbol("#push_trajectory!#9")),NamedTuple{(:t, :primal, :dual, :dual_gap, :time, :cache_size, :x, :v, :gamma), Tuple{Int64, Float64, Float64, Float64, Float64, Int64, Matrix{Float64}, RankOneMatrix{Float64, Vector{Float64}, Vector{Float64}}, Float64}}})   # time: 0.016703928
    Base.precompile(Tuple{Type{MultiCacheLMO{_A, NuclearNormLMO{Float64}, _B}} where {_A, _B},NuclearNormLMO{Float64}})   # time: 0.012803754
    Base.precompile(Tuple{typeof(print_callback),Tuple{String, String, Float64, Any, Any, Float64, Float64, Any},String})   # time: 0.011920895
    Base.precompile(Tuple{Type{VectorCacheLMO{NuclearNormLMO{Float64}, _A}} where _A,NuclearNormLMO{Float64}})   # time: 0.011098305
    Base.precompile(Tuple{typeof(print_callback),Tuple{String, String, Float64, Float64, Float64, Float64, Float64, Int64},String})   # time: 0.009940316
    isdefined(FrankWolfe, Symbol("#push_trajectory!#9")) && Base.precompile(Tuple{getfield(FrankWolfe, Symbol("#push_trajectory!#9")),NamedTuple{(:t, :primal, :dual, :dual_gap, :time, :cache_size, :x, :v, :gamma), _A} where _A<:Tuple{Int64, Float64, Any, Any, Float64, Any, Union{RankOneMatrix{Float64, Vector{Float64}, Vector{Float64}}, Matrix{Float64}}, Any, Any}})   # time: 0.005859037
    Base.precompile(Tuple{typeof(print_callback),Tuple{String, String, Any, Any, Any, Float64, Float64, Any},String})   # time: 0.003590876
    Base.precompile(Tuple{Core.kwftype(typeof(print_callback)),NamedTuple{(:print_header,), Tuple{Bool}},typeof(print_callback),Vector{String},String})   # time: 0.001366999
end
