function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{Core.kwftype(typeof(lazified_conditional_gradient)),NamedTuple{(:max_iteration, :L, :line_search, :print_iter, :emphasis, :cache_size, :verbose), Tuple{Int64, Int64, Adaptive, Float64, Emphasis, Int64, Bool}},typeof(lazified_conditional_gradient),Function,Function,KSparseLMO{Float64},SparseVector{Float64, Int64}})   # time: 0.9227771
    Base.precompile(Tuple{Core.kwftype(typeof(compute_extreme_point)),NamedTuple{(:threshold, :greedy), Tuple{Float64, Bool}},typeof(compute_extreme_point),MultiCacheLMO{500, KSparseLMO{Float64}, SparseVector{Float64, Int64}},SparseVector{Float64, Int64}})   # time: 0.25742963
    Base.precompile(Tuple{typeof(length),MultiCacheLMO{500, KSparseLMO{Float64}, SparseVector{Float64, Int64}}})   # time: 0.036646128
    Base.precompile(Tuple{Type{MultiCacheLMO{500, KSparseLMO{Float64}, SparseVector{Float64, Int64}}},KSparseLMO{Float64}})   # time: 0.03627382
    Base.precompile(Tuple{Core.kwftype(typeof(compute_extreme_point)),NamedTuple{(:threshold, :greedy), Tuple{Float64, Bool}},typeof(compute_extreme_point),MultiCacheLMO{_A, KSparseLMO{Float64}, _B} where {_A, _B},SparseVector{Float64, Int64}})   # time: 0.022596858
    Base.precompile(Tuple{Core.kwftype(typeof(compute_extreme_point)),NamedTuple{(:threshold, :greedy), _A} where _A<:Tuple{Any, Bool},typeof(compute_extreme_point),MultiCacheLMO{_A, KSparseLMO{Float64}, _B} where {_A, _B},SparseVector{Float64, Int64}})   # time: 0.009330567
    Base.precompile(Tuple{typeof(print_callback),Tuple{String, String, Float64, Any, Any, Float64, Float64, Int64},String})   # time: 0.004769557
    Base.precompile(Tuple{typeof(length),MultiCacheLMO{_A, KSparseLMO{Float64}, _B} where {_A, _B}})   # time: 0.00342123
    isdefined(FrankWolfe, Symbol("#push_trajectory!#9")) && Base.precompile(Tuple{getfield(FrankWolfe, Symbol("#push_trajectory!#9")),NamedTuple{(:t, :primal, :dual, :dual_gap, :time, :cache_size, :x, :v, :gamma), _A} where _A<:Tuple{Int64, Float64, Any, Any, Float64, Int64, SparseVector{Float64, Int64}, Any, Any}})   # time: 0.002330826
end
