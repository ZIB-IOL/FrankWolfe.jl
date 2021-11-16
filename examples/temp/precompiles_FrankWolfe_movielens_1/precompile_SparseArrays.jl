function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(setindex!),SparseMatrixCSC{Float64, Int64},Any,Int64,Int64})   # time: 0.004553643
    Base.precompile(Tuple{typeof(getindex),SparseMatrixCSC{Float64, Int64},Int64,Int64})   # time: 0.001167811
end
