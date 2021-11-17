function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    isdefined(SparseArrays.HigherOrderFns, Symbol("#3#4")) && Base.precompile(Tuple{getfield(SparseArrays.HigherOrderFns, Symbol("#3#4")),Float64,Float64})   # time: 0.034728143
    Base.precompile(Tuple{typeof(_sparsifystructured),Vector{BigFloat}})   # time: 0.015891558
    Base.precompile(Tuple{typeof(_sparsifystructured),Vector{Float64}})   # time: 0.007754904
    Base.precompile(Tuple{typeof(_allocres),Tuple{Int64},Type{Int64},Type{Float64},Int64})   # time: 0.001293659
end
