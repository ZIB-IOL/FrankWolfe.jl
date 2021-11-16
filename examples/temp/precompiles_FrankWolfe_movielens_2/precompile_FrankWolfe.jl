function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(line_search_wrapper),Adaptive,Int64,Function,Function,Any,Any,SparseArrays.SparseMatrixCSC{Float64, Int64},Any,Any,Int64,Float64,Int64,Float64})   # time: 0.45071155
    Base.precompile(Tuple{Core.kwftype(typeof(compute_extreme_point)),NamedTuple{(:threshold, :greedy), Tuple{Float64, Bool}},typeof(compute_extreme_point),MultiCacheLMO{_A, NuclearNormLMO{Float64}, RankOneMatrix{Float64, Vector{Float64}, Vector{Float64}}} where _A,SparseArrays.SparseMatrixCSC{Float64, Int64}})   # time: 0.022829248
    Base.precompile(Tuple{Type{MultiCacheLMO{_A, NuclearNormLMO{Float64}, RankOneMatrix{Float64, Vector{Float64}, Vector{Float64}}}} where _A,NuclearNormLMO{Float64}})   # time: 0.007260614
    Base.precompile(Tuple{Type{MultiCacheLMO{_A, NuclearNormLMO{Float64}, _B}} where {_A, _B},NuclearNormLMO{Float64}})   # time: 0.007219895
    Base.precompile(Tuple{Core.kwftype(typeof(compute_extreme_point)),NamedTuple{(:threshold, :greedy), _A} where _A<:Tuple{Any, Bool},typeof(compute_extreme_point),VectorCacheLMO{NuclearNormLMO{Float64}, RankOneMatrix{Float64, Vector{Float64}, Vector{Float64}}},SparseArrays.SparseMatrixCSC{Float64, Int64}})   # time: 0.006727726
    Base.precompile(Tuple{Type{VectorCacheLMO{NuclearNormLMO{Float64}, _A}} where _A,NuclearNormLMO{Float64}})   # time: 0.00658588
    Base.precompile(Tuple{Core.kwftype(typeof(compute_extreme_point)),NamedTuple{(:threshold, :greedy), Tuple{Float64, Bool}},typeof(compute_extreme_point),VectorCacheLMO{NuclearNormLMO{Float64}, RankOneMatrix{Float64, Vector{Float64}, Vector{Float64}}},SparseArrays.SparseMatrixCSC{Float64, Int64}})   # time: 0.006375383
    Base.precompile(Tuple{Core.kwftype(typeof(compute_extreme_point)),NamedTuple{(:threshold, :greedy), _A} where _A<:Tuple{Any, Bool},typeof(compute_extreme_point),MultiCacheLMO{_A, NuclearNormLMO{Float64}, RankOneMatrix{Float64, Vector{Float64}, Vector{Float64}}} where _A,SparseArrays.SparseMatrixCSC{Float64, Int64}})   # time: 0.005405074
    Base.precompile(Tuple{typeof(length),MultiCacheLMO{_A, NuclearNormLMO{Float64}, RankOneMatrix{Float64, Vector{Float64}, Vector{Float64}}} where _A})   # time: 0.005332928
    Base.precompile(Tuple{typeof(print_callback),Tuple{String, String, Float64, Any, Any, Float64, Float64, Any},String})   # time: 0.004790049
    Base.precompile(Tuple{typeof(print_callback),Tuple{String, String, Any, Any, Any, Float64, Float64, Any},String})   # time: 0.001412537
end
