function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(copyto!),SparseVector{Float64, Int64},SparseVector{BigFloat, Int64}})   # time: 0.006425299
end
