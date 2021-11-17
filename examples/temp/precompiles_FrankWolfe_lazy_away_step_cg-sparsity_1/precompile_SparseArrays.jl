function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(getindex),SparseMatrixCSC{Float64, Int64},Vector{Int64},Vector{Int64}})   # time: 0.04750791
    Base.precompile(Tuple{typeof(-),SparseVector{Float64, Int64},SparseVector{Float64, Int64}})   # time: 0.02923603
    Base.precompile(Tuple{Type{SparseMatrixCSC},Int64,Int64,Vector{Int64},Vector{Int64},Vector{Float64}})   # time: 0.008432161
    Base.precompile(Tuple{typeof(dot),SparseVector{Float64, Int64},SparseVector{Float64, Int64}})   # time: 0.006709141
    Base.precompile(Tuple{typeof(getindex),SparseVector{Float64, Int64},UnitRange{Int64}})   # time: 0.002652049
end
