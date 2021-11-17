function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(widelength),SparseVector{Float64, Int64}})   # time: 0.011589281
    Base.precompile(Tuple{typeof(dot),SparseVector{Float64, Int64},SparseVector{Float64, Int64}})   # time: 0.009868834
end
