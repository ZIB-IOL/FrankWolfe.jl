function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(norm),SparseVector{Rational{BigInt}, Int64}})   # time: 0.046348583
    Base.precompile(Tuple{typeof(widelength),SparseVector{Rational{BigInt}, Int64}})   # time: 0.004984063
end
