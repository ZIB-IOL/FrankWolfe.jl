function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(print_callback),Tuple{String, String, Float64, Any, Any, Float64, Float64},String})   # time: 0.08973999
    Base.precompile(Tuple{typeof(fast_dot),SparseArrays.SparseMatrixCSC{BigFloat, Int64},Matrix{T} where T})   # time: 0.010672705
    Base.precompile(Tuple{Core.kwftype(typeof(print_callback)),NamedTuple{(:print_header,), Tuple{Bool}},typeof(print_callback),Vector{String},String})   # time: 0.001195119
end
