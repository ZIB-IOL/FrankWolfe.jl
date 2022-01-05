function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{Core.kwftype(typeof(lazified_conditional_gradient)),NamedTuple{(:max_iteration, :line_search, :print_iter, :epsilon, :memory_mode, :trajectory, :cache_size, :verbose), Tuple{Int64, Adaptive{Float64, Int64}, Float64, Float64, InplaceEmphasis, Bool, Int64, Bool}},typeof(lazified_conditional_gradient),Function,Function,BirkhoffPolytopeLMO,SparseArrays.SparseMatrixCSC{Float64, Int64}})   # time: 0.15854251
    Base.precompile(Tuple{Core.kwftype(typeof(compute_extreme_point)),NamedTuple{(:threshold, :greedy), Tuple{Float64, Bool}},typeof(compute_extreme_point),MultiCacheLMO{_A, BirkhoffPolytopeLMO} where _A,SparseArrays.SparseMatrixCSC{Float64, Int64}})   # time: 0.031452376
    Base.precompile(Tuple{typeof(length),MultiCacheLMO{_A, BirkhoffPolytopeLMO} where _A})   # time: 0.01073467
    Base.precompile(Tuple{Core.kwftype(typeof(compute_extreme_point)),NamedTuple{(:threshold, :greedy), _A} where _A<:Tuple{Any, Bool},typeof(compute_extreme_point),MultiCacheLMO{_A, BirkhoffPolytopeLMO} where _A,SparseArrays.SparseMatrixCSC{Float64, Int64}})   # time: 0.010322329
    Base.precompile(Tuple{Core.kwftype(typeof(print_callback)),NamedTuple{(:print_header,), Tuple{Bool}},typeof(print_callback),Vector{String},String})   # time: 0.003367443
    Base.precompile(Tuple{Core.kwftype(typeof(compute_extreme_point)),NamedTuple{(:threshold, :greedy), Tuple{Float64, Bool}},typeof(compute_extreme_point),MultiCacheLMO{500, BirkhoffPolytopeLMO, SparseArrays.SparseMatrixCSC{Float64, Int64}},SparseArrays.SparseMatrixCSC{Float64, Int64}})   # time: 0.003286707
end
