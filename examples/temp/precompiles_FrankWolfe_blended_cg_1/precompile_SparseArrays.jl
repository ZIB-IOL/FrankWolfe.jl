function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(-),SparseVector{BigFloat, Int64},SparseVector{BigFloat, Int64}})   # time: 0.049439546
    Base.precompile(Tuple{typeof(*),BigFloat,SparseVector{BigFloat, Int64}})   # time: 0.015967134
    Base.precompile(Tuple{typeof(*),BigFloat,SparseVector{Float64, Int64}})   # time: 0.01531586
    Base.precompile(Tuple{typeof(*),Float64,SparseVector{BigFloat, Int64}})   # time: 0.01374902
    Base.precompile(Tuple{typeof(norm),SparseVector{Float64, Int64}})   # time: 0.011405142
    Base.precompile(Tuple{typeof(*),Float64,SparseVector{Float64, Int64}})   # time: 0.009298656
    Base.precompile(Tuple{typeof(norm),SparseVector{BigFloat, Int64}})   # time: 0.008981885
    Base.precompile(Tuple{typeof(dot),SparseVector{Float64, Int64},SparseVector{Float64, Int64}})   # time: 0.008047174
    Base.precompile(Tuple{typeof(*),Transpose{Float64, Matrix{Float64}},SparseVector{Float64, Int64}})   # time: 0.006658766
    Base.precompile(Tuple{typeof(-),SparseVector{Float64, Int64},SparseVector{BigFloat, Int64}})   # time: 0.006592954
    Base.precompile(Tuple{typeof(-),SparseVector{BigFloat, Int64},SparseVector{Float64, Int64}})   # time: 0.006370019
    Base.precompile(Tuple{typeof(*),Matrix{Float64},SparseVector{Float64, Int64}})   # time: 0.005735407
    Base.precompile(Tuple{typeof(-),SparseVector{Float64, Int64},SparseVector{Float64, Int64}})   # time: 0.004980768
    Base.precompile(Tuple{typeof(copyto!),SparseVector{Float64, Int64},SparseVector{Float64, Int64}})   # time: 0.004975597
    Base.precompile(Tuple{typeof(copyto!),SparseVector{Float64, Int64},Vector{Float64}})   # time: 0.004831198
    Base.precompile(Tuple{typeof(widelength),SparseVector{Float64, Int64}})   # time: 0.004683078
    Base.precompile(Tuple{typeof(setindex!),SparseVector{Float64, Int64},Float64,Int64})   # time: 0.002824855
    Base.precompile(Tuple{typeof(dot),Vector{Float64},SparseVector{Float64, Int64}})   # time: 0.001689177
end
