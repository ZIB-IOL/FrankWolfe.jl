function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(-),SparseVector{BigFloat, Int64},SparseVector{BigFloat, Int64}})   # time: 0.006299416
    Base.precompile(Tuple{typeof(-),SparseVector{Float64, Int64},SparseVector{BigFloat, Int64}})   # time: 0.005910284
    Base.precompile(Tuple{typeof(-),SparseVector{BigFloat, Int64},SparseVector{Float64, Int64}})   # time: 0.005705317
end
