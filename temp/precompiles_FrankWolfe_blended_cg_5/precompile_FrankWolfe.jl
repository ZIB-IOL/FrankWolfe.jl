function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{Core.kwftype(typeof(blended_conditional_gradient)),NamedTuple{(:epsilon, :max_iteration, :line_search, :print_iter, :memory_mode, :verbose, :trajectory, :lazy_tolerance, :weight_purge_threshold), Tuple{Float64, Int64, Adaptive{Float64, Int64}, Float64, InplaceEmphasis, Bool, Bool, Float64, Float64}},typeof(blended_conditional_gradient),Function,Function,KSparseLMO{Float64},SparseVector{Float64, Int64}})   # time: 0.38173005
    isdefined(FrankWolfe, Symbol("#push_trajectory!#33")) && Base.precompile(Tuple{getfield(FrankWolfe, Symbol("#push_trajectory!#33")),NamedTuple{(:t, :primal, :dual, :dual_gap, :time, :x, :active_set_length, :non_simplex_iter), Tuple{Int64, Float64, Float64, Float64, Float64, SparseVector{Float64, Int64}, Int64, Int64}}})   # time: 0.05380722
    Base.precompile(Tuple{Core.kwftype(typeof(print_callback)),NamedTuple{(:print_header,), Tuple{Bool}},typeof(print_callback),NTuple{9, String},String})   # time: 0.003759396
end
