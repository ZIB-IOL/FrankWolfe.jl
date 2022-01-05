function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{Core.kwftype(typeof(frank_wolfe)),NamedTuple{(:max_iteration, :line_search, :print_iter, :memory_mode, :verbose, :epsilon, :trajectory), Tuple{Int64, Adaptive{Float64, Int64}, Float64, InplaceEmphasis, Bool, Float64, Bool}},typeof(frank_wolfe),Function,Function,KSparseLMO{Float64},SparseVector{Float64, Int64}})   # time: 0.26637408
    isdefined(FrankWolfe, Symbol("#push_trajectory!#33")) && Base.precompile(Tuple{getfield(FrankWolfe, Symbol("#push_trajectory!#33")),NamedTuple{(:t, :primal, :dual, :dual_gap, :time, :x, :v, :gamma), Tuple{Int64, Float64, Float64, Float64, Float64, SparseVector{Float64, Int64}, SparseVector{Float64, Int64}, Float64}}})   # time: 0.08290658
    Base.precompile(Tuple{Core.kwftype(typeof(Type)),NamedTuple{(:L_est,), Tuple{Float64}},Type{Adaptive}})   # time: 0.00490745
end
