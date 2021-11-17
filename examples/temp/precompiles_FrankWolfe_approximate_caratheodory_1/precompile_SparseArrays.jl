function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(*),Rational{Int64},SparseVector{Rational{BigInt}, Int64}})   # time: 0.022460762
    Base.precompile(Tuple{typeof(dot),SparseVector{Rational{BigInt}, Int64},SparseVector{Rational{BigInt}, Int64}})   # time: 0.010399815
    Base.precompile(Tuple{typeof(-),SparseVector{Rational{BigInt}, Int64},SparseVector{Rational{BigInt}, Int64}})   # time: 0.005640903
    Base.precompile(Tuple{typeof(setindex!),SparseVector{Rational{BigInt}, Int64},Rational{BigInt},Int64})   # time: 0.003451709
end
