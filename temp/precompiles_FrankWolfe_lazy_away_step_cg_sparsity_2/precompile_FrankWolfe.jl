function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(active_set_update!),ActiveSet{SparseVector{Float64, Int64}, Float64, SparseVector{Float64, Int64}},Float64,SparseVector{Float64, Int64},Bool,Nothing})   # time: 0.30240533
    Base.precompile(Tuple{Type{ActiveSet},Vector{Tuple{Float64, SparseVector{Float64, Int64}}}})   # time: 0.09152829
    Base.precompile(Tuple{typeof(perform_line_search),Adaptive{Float64, Int64},Int64,Function,Function,SparseVector{Float64, Int64},SparseVector{Float64, Int64},Vector{Float64},Float64,SparseVector{Float64, Int64}})   # time: 0.07034466
    Base.precompile(Tuple{Core.kwftype(typeof(lazy_afw_step)),NamedTuple{(:lazy_tolerance,), Tuple{Float64}},typeof(lazy_afw_step),SparseVector{Float64, Int64},SparseVector{Float64, Int64},KSparseLMO{Float64},ActiveSet{SparseVector{Float64, Int64}, Float64, SparseVector{Float64, Int64}},Float64})   # time: 0.032810967
    Base.precompile(Tuple{Core.kwftype(typeof(print_callback)),NamedTuple{(:print_header,), Tuple{Bool}},typeof(print_callback),NTuple{8, String},String})   # time: 0.015512193
    Base.precompile(Tuple{typeof(perform_line_search),Adaptive{Float64, Int64},Int64,Function,Function,SparseVector{Float64, Int64},SparseVector{Float64, Int64},SparseVector{Float64, Int64},Float64,SparseVector{Float64, Int64}})   # time: 0.006093831
    Base.precompile(Tuple{typeof(active_set_update!),ActiveSet{SparseVector{Float64, Int64}, Float64, SparseVector{Float64, Int64}},Float64,SparseVector{Float64, Int64},Bool,Int64})   # time: 0.005901421
    Base.precompile(Tuple{typeof(print_callback),Tuple{String, String, Any, Any, Float64, Float64, Float64, Int64},String})   # time: 0.003269429
    Base.precompile(Tuple{typeof(afw_step),SparseVector{Float64, Int64},SparseVector{Float64, Int64},KSparseLMO{Float64},ActiveSet{SparseVector{Float64, Int64}, Float64, SparseVector{Float64, Int64}}})   # time: 0.002024735
end
