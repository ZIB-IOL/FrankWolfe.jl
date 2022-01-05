function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(dot),SparseVector{Float64, Int64},SparseVector{Float64, Int64}})   # time: 0.011920913
    Base.precompile(Tuple{typeof(widelength),SparseVector{Float64, Int64}})   # time: 0.010482818
    Base.precompile(Tuple{typeof(*),Int64,SparseVector{Float64, Int64}})   # time: 0.001334324
end
