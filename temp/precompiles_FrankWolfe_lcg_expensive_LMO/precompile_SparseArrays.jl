function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(dot),SparseMatrixCSC{Float64, Int64},SparseMatrixCSC{Float64, Int64}})   # time: 0.005275491
end
