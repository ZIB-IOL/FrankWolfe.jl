function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(_sparsifystructured),Vector{BigFloat}})   # time: 0.005777536
    isdefined(SparseArrays.HigherOrderFns, Symbol("#3#4")) && Base.precompile(Tuple{getfield(SparseArrays.HigherOrderFns, Symbol("#3#4")),Float64,Float64})   # time: 0.002408822
    Base.precompile(Tuple{typeof(_iszero),BigFloat})   # time: 0.001009417
end
