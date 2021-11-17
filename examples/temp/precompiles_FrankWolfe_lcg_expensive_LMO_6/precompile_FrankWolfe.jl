function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{Core.kwftype(typeof(blended_conditional_gradient)),NamedTuple{(:max_iteration, :line_search, :print_iter, :epsilon, :linesearch_tol, :emphasis, :trajectory, :verbose), Tuple{Int64, Adaptive, Float64, Float64, Float64, Emphasis, Bool, Bool}},typeof(blended_conditional_gradient),Function,Function,BirkhoffPolytopeLMO,SparseArrays.SparseMatrixCSC{Float64, Int64}})   # time: 0.007375554
    Base.precompile(Tuple{Core.kwftype(typeof(print_callback)),NamedTuple{(:print_header,), Tuple{Bool}},typeof(print_callback),NTuple{9, String},String})   # time: 0.006770243
    Base.precompile(Tuple{Core.kwftype(typeof(lp_separation_oracle)),NamedTuple{(:inplace_loop, :force_fw_step), Tuple{Bool, Bool}},typeof(lp_separation_oracle),BirkhoffPolytopeLMO,ActiveSet{SparseArrays.SparseMatrixCSC{Float64, Int64}, Float64, SparseArrays.SparseMatrixCSC{Float64, Int64}},SparseArrays.SparseMatrixCSC{Float64, Int64},Float64,Float64})   # time: 0.001511832
end
