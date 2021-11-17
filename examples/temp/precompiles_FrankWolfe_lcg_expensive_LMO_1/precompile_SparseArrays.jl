function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(norm),SparseMatrixCSC{BigFloat, Int64}})   # time: 0.05950499
    Base.precompile(Tuple{typeof(norm),SparseMatrixCSC{Float64, Int64}})   # time: 0.031898715
    Base.precompile(Tuple{typeof(-),SparseMatrixCSC{BigFloat, Int64},SparseMatrixCSC{BigFloat, Int64}})   # time: 0.008765324
    Base.precompile(Tuple{typeof(-),SparseMatrixCSC{BigFloat, Int64},SparseMatrixCSC{Float64, Int64}})   # time: 0.00855769
    Base.precompile(Tuple{typeof(-),SparseMatrixCSC{Float64, Int64},SparseMatrixCSC{BigFloat, Int64}})   # time: 0.008427794
    Base.precompile(Tuple{typeof(dot),SparseMatrixCSC{Float64, Int64},SparseMatrixCSC{Float64, Int64}})   # time: 0.003426355
    Base.precompile(Tuple{typeof(similar),SparseMatrixCSC{Float64, Int64},Type{Float64}})   # time: 0.001382948
end
