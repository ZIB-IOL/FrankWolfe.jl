function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(getindex),SparseMatrixCSC{Float64, Int64},Vector{Int64},Vector{Int64}})   # time: 0.050151777
    Base.precompile(Tuple{typeof(-),SparseVector{Float64, Int64},SparseVector{BigFloat, Int64}})   # time: 0.03425098
    Base.precompile(Tuple{Type{SparseMatrixCSC},Int64,Int64,Vector{Int64},Vector{Int64},Vector{Float64}})   # time: 0.029823745
    Base.precompile(Tuple{typeof(*),BigFloat,SparseVector{Float64, Int64}})   # time: 0.022703381
    Base.precompile(Tuple{typeof(*),Float64,SparseVector{BigFloat, Int64}})   # time: 0.022456352
    Base.precompile(Tuple{typeof(*),BigFloat,SparseVector{BigFloat, Int64}})   # time: 0.022395736
    Base.precompile(Tuple{typeof(norm),SparseVector{BigFloat, Int64}})   # time: 0.017149001
    Base.precompile(Tuple{typeof(dot),SparseVector{Float64, Int64},SparseVector{Float64, Int64}})   # time: 0.007172737
    Base.precompile(Tuple{typeof(-),SparseVector{BigFloat, Int64},SparseVector{Float64, Int64}})   # time: 0.007124751
    Base.precompile(Tuple{typeof(-),SparseVector{BigFloat, Int64},SparseVector{BigFloat, Int64}})   # time: 0.006985773
    Base.precompile(Tuple{typeof(-),SparseVector{Float64, Int64},SparseVector{Float64, Int64}})   # time: 0.005108188
    Base.precompile(Tuple{typeof(getindex),SparseVector{Float64, Int64},UnitRange{Int64}})   # time: 0.00340583
end
