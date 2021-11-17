function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(_sparsifystructured),Vector{BigFloat}})   # time: 0.012885587
    isdefined(SparseArrays.HigherOrderFns, Symbol("#3#4")) && Base.precompile(Tuple{getfield(SparseArrays.HigherOrderFns, Symbol("#3#4")),Float64,Float64})   # time: 0.002893272
    Base.precompile(Tuple{typeof(_iszero),BigFloat})   # time: 0.001645015
    Base.precompile(Tuple{typeof(_allocres),Tuple{Int64},Type{Int64},Type{BigFloat},Int64})   # time: 0.001280589
end
