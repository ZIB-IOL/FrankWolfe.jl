function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{Core.kwftype(typeof(compute_extreme_point)),NamedTuple{(:threshold, :greedy), Tuple{Float64, Bool}},typeof(compute_extreme_point),MultiCacheLMO{_A, NuclearNormLMO{Float64}, RankOneMatrix{Float64, Vector{Float64}, Vector{Float64}}} where _A,SparseArrays.SparseMatrixCSC{Float64, Int64}})   # time: 0.029216602
    Base.precompile(Tuple{Core.kwftype(typeof(compute_extreme_point)),NamedTuple{(:threshold, :greedy), _A} where _A<:Tuple{Any, Bool},typeof(compute_extreme_point),VectorCacheLMO{NuclearNormLMO{Float64}, RankOneMatrix{Float64, Vector{Float64}, Vector{Float64}}},SparseArrays.SparseMatrixCSC{Float64, Int64}})   # time: 0.010391894
    Base.precompile(Tuple{Core.kwftype(typeof(compute_extreme_point)),NamedTuple{(:threshold, :greedy), _A} where _A<:Tuple{Any, Bool},typeof(compute_extreme_point),MultiCacheLMO{_A, NuclearNormLMO{Float64}, RankOneMatrix{Float64, Vector{Float64}, Vector{Float64}}} where _A,SparseArrays.SparseMatrixCSC{Float64, Int64}})   # time: 0.005250637
    Base.precompile(Tuple{typeof(length),MultiCacheLMO{_A, NuclearNormLMO{Float64}, RankOneMatrix{Float64, Vector{Float64}, Vector{Float64}}} where _A})   # time: 0.004705316
    Base.precompile(Tuple{typeof(print_callback),Tuple{String, String, Any, Any, Any, Float64, Float64, Int64},String})   # time: 0.00435821
    Base.precompile(Tuple{typeof(print_callback),Tuple{String, String, Float64, Any, Any, Float64, Float64, Any},String})   # time: 0.003944127
    Base.precompile(Tuple{Type{MultiCacheLMO{_A, NuclearNormLMO{Float64}, _B}} where {_A, _B},NuclearNormLMO{Float64}})   # time: 0.00359918
    Base.precompile(Tuple{Type{MultiCacheLMO{_A, NuclearNormLMO{Float64}, RankOneMatrix{Float64, Vector{Float64}, Vector{Float64}}}} where _A,NuclearNormLMO{Float64}})   # time: 0.003546448
    Base.precompile(Tuple{Type{VectorCacheLMO{NuclearNormLMO{Float64}, _A}} where _A,NuclearNormLMO{Float64}})   # time: 0.003143188
    Base.precompile(Tuple{Core.kwftype(typeof(compute_extreme_point)),NamedTuple{(:threshold, :greedy), Tuple{Float64, Bool}},typeof(compute_extreme_point),VectorCacheLMO{NuclearNormLMO{Float64}, RankOneMatrix{Float64, Vector{Float64}, Vector{Float64}}},SparseArrays.SparseMatrixCSC{Float64, Int64}})   # time: 0.002158991
    Base.precompile(Tuple{Core.kwftype(typeof(print_callback)),NamedTuple{(:print_header,), Tuple{Bool}},typeof(print_callback),Vector{String},String})   # time: 0.001511499
end
