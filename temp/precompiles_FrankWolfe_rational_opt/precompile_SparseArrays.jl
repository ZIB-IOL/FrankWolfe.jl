function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(copyto!),SparseVector{Rational{BigInt}, Int64},SparseVector{Rational{BigInt}, Int64}})   # time: 0.01076556
    Base.precompile(Tuple{typeof(dot),SparseVector{BigFloat, Int64},SparseVector{BigFloat, Int64}})   # time: 0.009790923
end
