function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(*),Matrix{Float64},SparseVector{Float64, Int64}})   # time: 0.005172924
end
