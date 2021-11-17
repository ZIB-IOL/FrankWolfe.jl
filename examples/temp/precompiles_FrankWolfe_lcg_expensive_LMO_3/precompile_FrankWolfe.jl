function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{Core.kwftype(typeof(compute_extreme_point)),NamedTuple{(:threshold, :greedy), Tuple{Float64, Bool}},typeof(compute_extreme_point),MultiCacheLMO{500, BirkhoffPolytopeLMO, SparseArrays.SparseMatrixCSC{Float64, Int64}},SparseArrays.SparseMatrixCSC{Float64, Int64}})   # time: 0.25589186
    Base.precompile(Tuple{Core.kwftype(typeof(lazified_conditional_gradient)),NamedTuple{(:max_iteration, :line_search, :print_iter, :epsilon, :emphasis, :trajectory, :cache_size, :verbose), Tuple{Int64, Adaptive, Float64, Float64, Emphasis, Bool, Int64, Bool}},typeof(lazified_conditional_gradient),Function,Function,BirkhoffPolytopeLMO,SparseArrays.SparseMatrixCSC{Float64, Int64}})   # time: 0.060838394
    Base.precompile(Tuple{typeof(length),MultiCacheLMO{500, BirkhoffPolytopeLMO, SparseArrays.SparseMatrixCSC{Float64, Int64}}})   # time: 0.039786853
    Base.precompile(Tuple{Type{MultiCacheLMO{500, BirkhoffPolytopeLMO, SparseArrays.SparseMatrixCSC{Float64, Int64}}},BirkhoffPolytopeLMO})   # time: 0.036974385
    Base.precompile(Tuple{Core.kwftype(typeof(compute_extreme_point)),NamedTuple{(:threshold, :greedy), Tuple{Float64, Bool}},typeof(compute_extreme_point),MultiCacheLMO{_A, BirkhoffPolytopeLMO, _B} where {_A, _B},SparseArrays.SparseMatrixCSC{Float64, Int64}})   # time: 0.015717806
    Base.precompile(Tuple{Core.kwftype(typeof(compute_extreme_point)),NamedTuple{(:threshold, :greedy), _A} where _A<:Tuple{Any, Bool},typeof(compute_extreme_point),MultiCacheLMO{_A, BirkhoffPolytopeLMO, _B} where {_A, _B},SparseArrays.SparseMatrixCSC{Float64, Int64}})   # time: 0.005551617
    Base.precompile(Tuple{typeof(length),MultiCacheLMO{_A, BirkhoffPolytopeLMO, _B} where {_A, _B}})   # time: 0.003511508
end
