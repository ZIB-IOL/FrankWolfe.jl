function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{Core.kwftype(typeof(blended_conditional_gradient)),NamedTuple{(:epsilon, :max_iteration, :line_search, :print_iter, :emphasis, :L, :verbose, :trajectory, :K, :weight_purge_threshold), Tuple{Float64, Int64, Adaptive, Float64, Emphasis, Float64, Bool, Bool, Float64, Float64}},typeof(blended_conditional_gradient),Function,Function,KSparseLMO{Float64},SparseVector{Float64, Int64}})   # time: 0.13206175
    Base.precompile(Tuple{Core.kwftype(typeof(print_callback)),NamedTuple{(:print_header,), Tuple{Bool}},typeof(print_callback),NTuple{9, String},String})   # time: 0.003540223
end
