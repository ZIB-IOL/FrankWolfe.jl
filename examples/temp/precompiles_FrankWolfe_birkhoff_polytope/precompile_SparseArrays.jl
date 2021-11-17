function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(norm),SparseMatrixCSC{Float64, Int64}})   # time: 0.031131797
    Base.precompile(Tuple{typeof(norm),SparseMatrixCSC{BigFloat, Int64}})   # time: 0.030752905
    Base.precompile(Tuple{typeof(-),SparseMatrixCSC{BigFloat, Int64},SparseMatrixCSC{BigFloat, Int64}})   # time: 0.009913751
    Base.precompile(Tuple{typeof(-),SparseMatrixCSC{Float64, Int64},SparseMatrixCSC{BigFloat, Int64}})   # time: 0.00933718
    Base.precompile(Tuple{typeof(-),SparseMatrixCSC{BigFloat, Int64},SparseMatrixCSC{Float64, Int64}})   # time: 0.009289834
    Base.precompile(Tuple{typeof(dot),SparseMatrixCSC{Float64, Int64},SparseMatrixCSC{Float64, Int64}})   # time: 0.002958582
end
