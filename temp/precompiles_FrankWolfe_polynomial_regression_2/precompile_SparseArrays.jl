function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(widelength),SparseVector{Float64, Int64}})   # time: 0.03508232
    Base.precompile(Tuple{typeof(dot),SparseVector{Float64, Int64},SparseVector{Float64, Int64}})   # time: 0.006732865
    Base.precompile(Tuple{typeof(fill!),SparseVector{Float64, Int64},Int64})   # time: 0.001662978
end
