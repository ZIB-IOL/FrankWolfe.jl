function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{Core.kwftype(typeof(lazified_conditional_gradient)),NamedTuple{(:max_iteration, :L, :line_search, :print_iter, :emphasis, :verbose), Tuple{Int64, Int64, Adaptive, Float64, Emphasis, Bool}},typeof(lazified_conditional_gradient),Function,Function,KSparseLMO{Float64},SparseVector{Float64, Int64}})   # time: 3.9177554
    Base.precompile(Tuple{Type{MultiCacheLMO{_A, KSparseLMO{Float64}, _B}} where {_A, _B},KSparseLMO{Float64}})   # time: 0.012864536
    Base.precompile(Tuple{Type{VectorCacheLMO{KSparseLMO{Float64}, _A}} where _A,KSparseLMO{Float64}})   # time: 0.011564771
    Base.precompile(Tuple{typeof(print_callback),Tuple{String, String, Float64, Any, Any, Float64, Float64, Any},String})   # time: 0.008668335
    Base.precompile(Tuple{typeof(print_callback),Tuple{String, String, Float64, Float64, Float64, Float64, Float64, Int64},String})   # time: 0.005350467
    Base.precompile(Tuple{typeof(print_callback),Tuple{String, String, Any, Any, Any, Float64, Float64, Any},String})   # time: 0.002514951
end
