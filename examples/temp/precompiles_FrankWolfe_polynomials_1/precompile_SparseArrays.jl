function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(prep_sparsevec_copy_dest!),SparseVector{Float64, Int64},Int64,Int64})   # time: 0.00683922
    Base.precompile(Tuple{typeof(dot),SparseVector{BigFloat, Int64},Vector{Float64}})   # time: 0.005440938
    Base.precompile(Tuple{typeof(dot),SparseVector{Float64, Int64},Vector{Float64}})   # time: 0.002786016
    Base.precompile(Tuple{typeof(similar),SparseVector{Float64, Int64},Type{Float64}})   # time: 0.002758426
end
