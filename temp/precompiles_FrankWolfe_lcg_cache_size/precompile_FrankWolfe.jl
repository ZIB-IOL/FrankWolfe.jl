function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{Core.kwftype(typeof(lazified_conditional_gradient)),NamedTuple{(:max_iteration, :line_search, :print_iter, :memory_mode, :verbose), Tuple{Int64, Adaptive{Float64, Int64}, Float64, InplaceEmphasis, Bool}},typeof(lazified_conditional_gradient),Function,Function,KSparseLMO{Float64},SparseVector{Float64, Int64}})   # time: 0.71887785
    Base.precompile(Tuple{Core.kwftype(typeof(Type)),NamedTuple{(:L_est,), Tuple{Float64}},Type{Adaptive}})   # time: 0.009307879
    Base.precompile(Tuple{Type{MultiCacheLMO{_A, KSparseLMO{Float64}, _B}} where {_A, _B},KSparseLMO{Float64}})   # time: 0.007678914
    Base.precompile(Tuple{typeof(print_callback),Tuple{String, String, Float64, Any, Any, Float64, Float64, Any},String})   # time: 0.007181576
    Base.precompile(Tuple{Type{VectorCacheLMO{KSparseLMO{Float64}, _A}} where _A,KSparseLMO{Float64}})   # time: 0.005332965
    isdefined(FrankWolfe, Symbol("#push_trajectory!#33")) && Base.precompile(Tuple{getfield(FrankWolfe, Symbol("#push_trajectory!#33")),NamedTuple{(:t, :primal, :dual, :dual_gap, :time, :cache_size, :x, :v, :gamma), _A} where _A<:Tuple{Int64, Float64, Any, Any, Float64, Any, SparseVector{Float64, Int64}, Any, Float64}})   # time: 0.002685098
    Base.precompile(Tuple{Core.kwftype(typeof(print_callback)),NamedTuple{(:print_header,), Tuple{Bool}},typeof(print_callback),Vector{String},String})   # time: 0.001523225
end
