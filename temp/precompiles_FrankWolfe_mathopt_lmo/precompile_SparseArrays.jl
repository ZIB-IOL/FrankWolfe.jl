function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(getindex),SparseVector{Float64, Int64},Int64})   # time: 0.002340579
end
