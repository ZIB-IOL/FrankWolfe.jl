function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(dot),SparseVector{BigFloat, Int64},SparseVector{BigFloat, Int64}})   # time: 0.00958512
end
