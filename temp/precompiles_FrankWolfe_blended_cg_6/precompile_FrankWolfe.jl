function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{Core.kwftype(typeof(blended_pairwise_conditional_gradient)),NamedTuple{(:epsilon, :max_iteration, :line_search, :print_iter, :memory_mode, :verbose, :trajectory), Tuple{Float64, Int64, Adaptive{Float64, Int64}, Float64, InplaceEmphasis, Bool, Bool}},typeof(blended_pairwise_conditional_gradient),Function,Function,KSparseLMO{Float64},SparseVector{Float64, Int64}})   # time: 0.18068784
    isdefined(FrankWolfe, Symbol("#push_trajectory!#33")) && Base.precompile(Tuple{getfield(FrankWolfe, Symbol("#push_trajectory!#33")),NamedTuple{(:t, :primal, :dual, :dual_gap, :time, :x, :v, :active_set_length, :gamma), Tuple{Int64, Float64, Float64, Float64, Float64, SparseVector{Float64, Int64}, SparseVector{Float64, Int64}, Int64, Float64}}})   # time: 0.027121874
    Base.precompile(Tuple{Core.kwftype(typeof(print_callback)),NamedTuple{(:print_header,), Tuple{Bool}},typeof(print_callback),NTuple{8, String},String})   # time: 0.005311987
end
