function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{Core.kwftype(typeof(lazified_conditional_gradient)),NamedTuple{(:max_iteration, :epsilon, :line_search, :print_iter, :memory_mode, :trajectory, :verbose), Tuple{Int64, Float64, Adaptive{Float64, Int64}, Float64, InplaceEmphasis, Bool, Bool}},typeof(lazified_conditional_gradient),Function,Function,BirkhoffPolytopeLMO,SparseArrays.SparseMatrixCSC{Float64, Int64}})   # time: 0.326898
    isdefined(FrankWolfe, Symbol("#push_trajectory!#33")) && Base.precompile(Tuple{getfield(FrankWolfe, Symbol("#push_trajectory!#33")),NamedTuple{(:t, :primal, :dual, :dual_gap, :time, :cache_size, :x, :v, :gamma), Tuple{Int64, Float64, Float64, Float64, Float64, Int64, SparseArrays.SparseMatrixCSC{Float64, Int64}, SparseArrays.SparseMatrixCSC{Float64, Int64}, Float64}}})   # time: 0.036940474
    Base.precompile(Tuple{typeof(print_callback),Tuple{String, String, Any, Any, Any, Float64, Float64, Any},String})   # time: 0.010050932
    Base.precompile(Tuple{Type{MultiCacheLMO{_A, BirkhoffPolytopeLMO, _B}} where {_A, _B},BirkhoffPolytopeLMO})   # time: 0.007305136
    Base.precompile(Tuple{Type{VectorCacheLMO{BirkhoffPolytopeLMO, _A}} where _A,BirkhoffPolytopeLMO})   # time: 0.005527968
    isdefined(FrankWolfe, Symbol("#push_trajectory!#33")) && Base.precompile(Tuple{getfield(FrankWolfe, Symbol("#push_trajectory!#33")),NamedTuple{(:t, :primal, :dual, :dual_gap, :time, :cache_size, :x, :v, :gamma), _A} where _A<:Tuple{Int64, Any, Any, Any, Float64, Any, SparseArrays.SparseMatrixCSC{Float64, Int64}, Any, Float64}})   # time: 0.004072484
    Base.precompile(Tuple{Core.kwftype(typeof(print_callback)),NamedTuple{(:print_header,), Tuple{Bool}},typeof(print_callback),Vector{String},String})   # time: 0.003188512
end
