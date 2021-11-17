function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(norm),SparseVector{Rational{BigInt}, Int64}})   # time: 0.020194769
    Base.precompile(Tuple{typeof(widelength),SparseVector{Rational{BigInt}, Int64}})   # time: 0.006099362
end
