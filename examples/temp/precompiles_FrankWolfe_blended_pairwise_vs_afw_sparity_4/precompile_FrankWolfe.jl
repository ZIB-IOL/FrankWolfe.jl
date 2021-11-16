function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(active_set_update_iterate_pairwise!),ActiveSet{SparseVector{Float64, Int64}, Float64, SparseVector{Float64, Int64}},Float64,SparseVector{Float64, Int64},SparseVector{Float64, Int64}})   # time: 0.08547271
    Base.precompile(Tuple{typeof(active_set_initialize!),ActiveSet{SparseVector{Float64, Int64}, Float64, SparseVector{Float64, Int64}},SparseVector{Float64, Int64}})   # time: 0.06373713
    Base.precompile(Tuple{typeof(active_set_update_iterate_pairwise!),ActiveSet{SparseVector{Float64, Int64}, Float64, SparseVector{Float64, Int64}},BigFloat,SparseVector{Float64, Int64},SparseVector{Float64, Int64}})   # time: 0.049117424
    Base.precompile(Tuple{typeof(active_set_update_iterate_pairwise!),ActiveSet{SparseVector{Float64, Int64}, Float64, SparseVector{Float64, Int64}},Int64,SparseVector{Float64, Int64},SparseVector{Float64, Int64}})   # time: 0.04757917
    Base.precompile(Tuple{Core.kwftype(typeof(print_callback)),NamedTuple{(:print_header,), Tuple{Bool}},typeof(print_callback),NTuple{8, String},String})   # time: 0.005582537
    Base.precompile(Tuple{typeof(line_search_wrapper),Adaptive,Int64,Function,Function,SparseVector{Float64, Int64},SparseVector{Float64, Int64},SparseVector{Float64, Int64},Float64,Float64,Int64,Float64,Int64,Float64})   # time: 0.001782136
end
