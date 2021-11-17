function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{Core.kwftype(typeof(away_frank_wolfe)),NamedTuple{(:max_iteration, :line_search, :print_iter, :linesearch_tol, :epsilon, :emphasis, :lazy, :trajectory, :verbose), Tuple{Int64, Adaptive, Float64, Float64, Float64, Emphasis, Bool, Bool, Bool}},typeof(away_frank_wolfe),Function,Function,BirkhoffPolytopeLMO,SparseArrays.SparseMatrixCSC{Float64, Int64}})   # time: 1.474222
    Base.precompile(Tuple{Core.kwftype(typeof(print_callback)),NamedTuple{(:print_header,), Tuple{Bool}},typeof(print_callback),NTuple{8, String},String})   # time: 0.012870593
    isdefined(FrankWolfe, Symbol("#push_trajectory!#9")) && Base.precompile(Tuple{getfield(FrankWolfe, Symbol("#push_trajectory!#9")),NamedTuple{(:t, :primal, :dual, :dual_gap, :time, :x, :v, :active_set_length, :gamma), Tuple{Int64, Float64, Float64, Float64, Float64, SparseArrays.SparseMatrixCSC{Float64, Int64}, SparseArrays.SparseMatrixCSC{Float64, Int64}, Int64, Float64}}})   # time: 0.012551221
end
