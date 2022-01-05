function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(_sparsifystructured),Vector{Float64}})   # time: 0.00333162
    Base.precompile(Tuple{typeof(_allocres),Tuple{Int64},Type{Int64},Type{BigFloat},Int64})   # time: 0.002414817
    Base.precompile(Tuple{typeof(_iszero),BigFloat})   # time: 0.001588597
    isdefined(SparseArrays.HigherOrderFns, Symbol("#3#4")) && Base.precompile(Tuple{getfield(SparseArrays.HigherOrderFns, Symbol("#3#4")),Float64,Float64})   # time: 0.001217257
end
