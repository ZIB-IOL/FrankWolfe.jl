function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{Core.kwftype(typeof(lazified_conditional_gradient)),NamedTuple{(:max_iteration, :epsilon, :line_search, :print_iter, :emphasis, :trajectory, :verbose), Tuple{Int64, Float64, Adaptive, Float64, Emphasis, Bool, Bool}},typeof(lazified_conditional_gradient),Function,Function,BirkhoffPolytopeLMO,SparseArrays.SparseMatrixCSC{Float64, Int64}})   # time: 1.0580478
    Base.precompile(Tuple{typeof(print_callback),Tuple{String, String, Any, Any, Any, Float64, Float64, Any},String})   # time: 0.04494054
    isdefined(FrankWolfe, Symbol("#push_trajectory!#9")) && Base.precompile(Tuple{getfield(FrankWolfe, Symbol("#push_trajectory!#9")),NamedTuple{(:t, :primal, :dual, :dual_gap, :time, :cache_size, :x, :v, :gamma), Tuple{Int64, Float64, Float64, Float64, Float64, Int64, SparseArrays.SparseMatrixCSC{Float64, Int64}, SparseArrays.SparseMatrixCSC{Float64, Int64}, Float64}}})   # time: 0.008244496
    Base.precompile(Tuple{Type{MultiCacheLMO{_A, BirkhoffPolytopeLMO, _B}} where {_A, _B},BirkhoffPolytopeLMO})   # time: 0.006924912
    Base.precompile(Tuple{Type{VectorCacheLMO{BirkhoffPolytopeLMO, _A}} where _A,BirkhoffPolytopeLMO})   # time: 0.00617293
    Base.precompile(Tuple{typeof(print_callback),Tuple{String, String, Float64, Float64, Float64, Float64, Float64, Int64},String})   # time: 0.005058614
    isdefined(FrankWolfe, Symbol("#push_trajectory!#9")) && Base.precompile(Tuple{getfield(FrankWolfe, Symbol("#push_trajectory!#9")),NamedTuple{(:t, :primal, :dual, :dual_gap, :time, :cache_size, :x, :v, :gamma), _A} where _A<:Tuple{Int64, Any, Any, Any, Float64, Any, SparseArrays.SparseMatrixCSC{Float64, Int64}, Any, Any}})   # time: 0.002354069
end
