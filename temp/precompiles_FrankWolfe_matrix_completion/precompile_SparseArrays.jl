function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(*),Adjoint{Float64, SparseMatrixCSC{Float64, Int64}},Matrix{Float64}})   # time: 0.005950723
end
