function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(getindex),SparseMatrixCSC{Float64, Int64},Vector{Int64},Vector{Int64}})   # time: 0.048785713
    Base.precompile(Tuple{typeof(getindex),SparseVector{Float64, Int64},UnitRange{Int64}})   # time: 0.002013161
end
