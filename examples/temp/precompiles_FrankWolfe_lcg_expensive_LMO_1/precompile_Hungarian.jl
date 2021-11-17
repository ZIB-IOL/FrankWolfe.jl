function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(munkres),SparseMatrixCSC{Float64, Int64}})   # time: 0.045113035
end
