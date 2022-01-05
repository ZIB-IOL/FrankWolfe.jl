function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(_sparsifystructured),Vector{Rational{BigInt}}})   # time: 0.007187613
    isdefined(SparseArrays.HigherOrderFns, Symbol("#3#4")) && Base.precompile(Tuple{getfield(SparseArrays.HigherOrderFns, Symbol("#3#4")),Rational{BigInt},Rational{BigInt}})   # time: 0.001769354
    Base.precompile(Tuple{typeof(_allocres),Tuple{Int64},Type{Int64},Type{Rational{BigInt}},Int64})   # time: 0.001186393
    isdefined(SparseArrays.HigherOrderFns, Symbol("#3#4")) && Base.precompile(Tuple{getfield(SparseArrays.HigherOrderFns, Symbol("#3#4")),Rational{BigInt},Rational{BigInt}})   # time: 0.001099931
end
