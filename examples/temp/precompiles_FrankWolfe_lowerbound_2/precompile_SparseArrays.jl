function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(-),SparseVector{BigFloat, Int64},SparseVector{BigFloat, Int64}})   # time: 0.014214366
    Base.precompile(Tuple{typeof(-),SparseVector{BigFloat, Int64},SparseVector{Float64, Int64}})   # time: 0.01331755
    Base.precompile(Tuple{typeof(-),SparseVector{Float64, Int64},SparseVector{BigFloat, Int64}})   # time: 0.012538831
end
