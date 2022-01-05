function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(active_set_update_iterate_pairwise!),ActiveSet{SparseVector{Float64, Int64}, Float64, SparseVector{Float64, Int64}},Float64,SparseVector{Float64, Int64},SparseVector{Float64, Int64}})   # time: 0.22706518
    Base.precompile(Tuple{typeof(active_set_initialize!),ActiveSet{SparseVector{Float64, Int64}, Float64, SparseVector{Float64, Int64}},SparseVector{Float64, Int64}})   # time: 0.07041422
    Base.precompile(Tuple{Core.kwftype(typeof(print_callback)),NamedTuple{(:print_header,), Tuple{Bool}},typeof(print_callback),NTuple{8, String},String})   # time: 0.012517997
    Base.precompile(Tuple{typeof(deleteat!),ActiveSet{SparseVector{Float64, Int64}, Float64, SparseVector{Float64, Int64}},Int64})   # time: 0.001469705
end
