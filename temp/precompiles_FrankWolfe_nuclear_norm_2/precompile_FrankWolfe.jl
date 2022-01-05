function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{Core.kwftype(typeof(lazified_conditional_gradient)),NamedTuple{(:epsilon, :max_iteration, :print_iter, :trajectory, :verbose, :line_search, :memory_mode, :gradient), Tuple{Float64, Int64, Float64, Bool, Bool, Adaptive{Float64, Int64}, InplaceEmphasis, SparseArrays.SparseMatrixCSC{Float64, Int64}}},typeof(lazified_conditional_gradient),Function,Function,NuclearNormLMO{Float64},RankOneMatrix{Float64, Vector{Float64}, Vector{Float64}}})   # time: 0.36729497
    isdefined(FrankWolfe, Symbol("#push_trajectory!#33")) && Base.precompile(Tuple{getfield(FrankWolfe, Symbol("#push_trajectory!#33")),NamedTuple{(:t, :primal, :dual, :dual_gap, :time, :cache_size, :x, :v, :gamma), Tuple{Int64, Float64, Float64, Float64, Float64, Int64, Matrix{Float64}, RankOneMatrix{Float64, Vector{Float64}, Vector{Float64}}, Float64}}})   # time: 0.027153203
    Base.precompile(Tuple{typeof(print_callback),Tuple{String, String, Float64, Any, Any, Float64, Float64, Any},String})   # time: 0.011228724
    Base.precompile(Tuple{Type{MultiCacheLMO{_A, NuclearNormLMO{Float64}, _B}} where {_A, _B},NuclearNormLMO{Float64}})   # time: 0.006926254
    Base.precompile(Tuple{Type{VectorCacheLMO{NuclearNormLMO{Float64}, _A}} where _A,NuclearNormLMO{Float64}})   # time: 0.005551367
    isdefined(FrankWolfe, Symbol("#push_trajectory!#33")) && Base.precompile(Tuple{getfield(FrankWolfe, Symbol("#push_trajectory!#33")),NamedTuple{(:t, :primal, :dual, :dual_gap, :time, :cache_size, :x, :v, :gamma), _A} where _A<:Tuple{Int64, Float64, Any, Any, Float64, Any, Matrix{Float64}, Any, Float64}})   # time: 0.003839359
    Base.precompile(Tuple{Core.kwftype(typeof(print_callback)),NamedTuple{(:print_header,), Tuple{Bool}},typeof(print_callback),Vector{String},String})   # time: 0.001459603
end
