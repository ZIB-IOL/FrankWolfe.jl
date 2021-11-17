function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(dot),SparseVector{Float64, Int64},SparseVector{Float64, Int64}})   # time: 0.01006236
    Base.precompile(Tuple{typeof(fill!),SparseVector{Float64, Int64},Int64})   # time: 0.002996184
end
