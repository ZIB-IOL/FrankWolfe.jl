function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(_sparsifystructured),Vector{Float64}})   # time: 0.012201478
    Base.precompile(Tuple{typeof(_allocres),Tuple{Int64},Type{Int64},Type{Float64},Int64})   # time: 0.002272227
end
