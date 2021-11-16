function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(getindex),SparseMatrixCSC{Float64, Int64},Vector{Int64},Vector{Int64}})   # time: 0.04722439
    Base.precompile(Tuple{Type{SparseMatrixCSC},Int64,Int64,Vector{Int64},Vector{Int64},Vector{Float64}})   # time: 0.008108414
    Base.precompile(Tuple{typeof(dot),SparseVector{Float64, Int64},SparseVector{Float64, Int64}})   # time: 0.006353791
    Base.precompile(Tuple{typeof(-),SparseVector{Float64, Int64},SparseVector{Float64, Int64}})   # time: 0.004619919
    Base.precompile(Tuple{typeof(getindex),SparseVector{Float64, Int64},UnitRange{Int64}})   # time: 0.002876338
end
