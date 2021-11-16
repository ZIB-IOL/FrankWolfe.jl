function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(-),SparseVector{BigFloat, Int64},SparseVector{Float64, Int64}})   # time: 0.006991375
    Base.precompile(Tuple{typeof(-),SparseVector{BigFloat, Int64},SparseVector{BigFloat, Int64}})   # time: 0.006964834
    Base.precompile(Tuple{typeof(-),SparseVector{Float64, Int64},SparseVector{BigFloat, Int64}})   # time: 0.00643453
end
