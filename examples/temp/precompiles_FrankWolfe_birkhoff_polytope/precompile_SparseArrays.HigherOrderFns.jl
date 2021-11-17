function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    isdefined(SparseArrays.HigherOrderFns, Symbol("#3#4")) && Base.precompile(Tuple{getfield(SparseArrays.HigherOrderFns, Symbol("#3#4")),Float64,Vararg{Float64, N} where N})   # time: 0.03238594
    Base.precompile(Tuple{typeof(_allocres),Tuple{Int64, Int64},Type{Int64},Type{BigFloat},Int64})   # time: 0.00466828
    isdefined(SparseArrays.HigherOrderFns, Symbol("#3#4")) && Base.precompile(Tuple{getfield(SparseArrays.HigherOrderFns, Symbol("#3#4")),Float64,Float64})   # time: 0.004642093
    Base.precompile(Tuple{typeof(_sparsifystructured),Matrix{Float64}})   # time: 0.002683457
    isdefined(SparseArrays.HigherOrderFns, Symbol("#3#4")) && Base.precompile(Tuple{getfield(SparseArrays.HigherOrderFns, Symbol("#3#4")),Float64,Float64})   # time: 0.002077966
    isdefined(SparseArrays.HigherOrderFns, Symbol("#3#4")) && Base.precompile(Tuple{getfield(SparseArrays.HigherOrderFns, Symbol("#3#4")),Float64,Float64,Float64})   # time: 0.00163868
    Base.precompile(Tuple{typeof(_capturescalars),Base.RefValue{typeof(^)},Int64,Base.RefValue{Val{2}}})   # time: 0.001564314
    isdefined(SparseArrays.HigherOrderFns, Symbol("#3#4")) && Base.precompile(Tuple{getfield(SparseArrays.HigherOrderFns, Symbol("#3#4")),Float64,Float64,Float64})   # time: 0.001085217
end
