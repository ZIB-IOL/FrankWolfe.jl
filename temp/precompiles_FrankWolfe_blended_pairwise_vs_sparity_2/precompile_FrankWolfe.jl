function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(active_set_update!),ActiveSet{SparseVector{Float64, Int64}, Float64, SparseVector{Float64, Int64}},Float64,SparseVector{Float64, Int64},Bool,Nothing})   # time: 0.3297556
    Base.precompile(Tuple{typeof(perform_line_search),Adaptive{Float64, Int64},Int64,Function,Function,SparseVector{Float64, Int64},SparseVector{Float64, Int64},Vector{Float64},Float64,SparseVector{Float64, Int64}})   # time: 0.0803171
    Base.precompile(Tuple{Type{ActiveSet},Vector{Tuple{Float64, SparseVector{Float64, Int64}}}})   # time: 0.06371654
    Base.precompile(Tuple{Core.kwftype(typeof(lazy_afw_step)),NamedTuple{(:lazy_tolerance,), Tuple{Float64}},typeof(lazy_afw_step),SparseVector{Float64, Int64},SparseVector{Float64, Int64},KSparseLMO{Float64},ActiveSet{SparseVector{Float64, Int64}, Float64, SparseVector{Float64, Int64}},Float64})   # time: 0.032522447
    Base.precompile(Tuple{Core.kwftype(typeof(print_callback)),NamedTuple{(:print_header,), Tuple{Bool}},typeof(print_callback),NTuple{8, String},String})   # time: 0.011844151
    Base.precompile(Tuple{typeof(perform_line_search),Adaptive{Float64, Int64},Int64,Function,Function,SparseVector{Float64, Int64},SparseVector{Float64, Int64},SparseVector{Float64, Int64},Float64,SparseVector{Float64, Int64}})   # time: 0.007260863
    Base.precompile(Tuple{typeof(active_set_update!),ActiveSet{SparseVector{Float64, Int64}, Float64, SparseVector{Float64, Int64}},Float64,SparseVector{Float64, Int64},Bool,Int64})   # time: 0.007143604
    Base.precompile(Tuple{Type{Adaptive}})   # time: 0.005058407
    Base.precompile(Tuple{typeof(print_callback),Tuple{String, String, Any, Any, Float64, Float64, Float64, Int64},String})   # time: 0.003918484
    Base.precompile(Tuple{typeof(afw_step),SparseVector{Float64, Int64},SparseVector{Float64, Int64},KSparseLMO{Float64},ActiveSet{SparseVector{Float64, Int64}, Float64, SparseVector{Float64, Int64}}})   # time: 0.002095744
end
