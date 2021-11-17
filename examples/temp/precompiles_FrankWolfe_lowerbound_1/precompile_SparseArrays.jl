function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(*),Rational{Int64},SparseVector{Float64, Int64}})   # time: 0.03167631
    Base.precompile(Tuple{typeof(dot),SparseVector{Float64, Int64},SparseVector{Float64, Int64}})   # time: 0.0173759
    Base.precompile(Tuple{typeof(widelength),SparseVector{Float64, Int64}})   # time: 0.012170631
    Base.precompile(Tuple{typeof(-),SparseVector{Float64, Int64},SparseVector{Float64, Int64}})   # time: 0.009477455
    Base.precompile(Tuple{typeof(setindex!),SparseVector{Float64, Int64},Float64,Int64})   # time: 0.006661567
    Base.precompile(Tuple{typeof(similar),SparseVector{Float64, Int64},Type{Float64}})   # time: 0.001339463
    Base.precompile(Tuple{typeof(getindex),SparseVector{Float64, Int64},Int64})   # time: 0.001131761
end
