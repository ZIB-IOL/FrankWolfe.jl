function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(getindex),SparseMatrixCSC{Float64, Int64},Vector{Int64},Vector{Int64}})   # time: 0.048293196
    Base.precompile(Tuple{typeof(dot),SparseVector{Float64, Int64},SparseVector{Float64, Int64}})   # time: 0.040231165
    Base.precompile(Tuple{Type{SparseMatrixCSC},Int64,Int64,Vector{Int64},Vector{Int64},Vector{Float64}})   # time: 0.008584572
    Base.precompile(Tuple{typeof(-),SparseVector{Float64, Int64},SparseVector{Float64, Int64}})   # time: 0.004711336
    Base.precompile(Tuple{typeof(getindex),SparseVector{Float64, Int64},UnitRange{Int64}})   # time: 0.002590126
end
