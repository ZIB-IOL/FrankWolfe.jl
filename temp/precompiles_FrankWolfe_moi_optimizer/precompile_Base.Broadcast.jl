function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(materialize!),Vector{Float64},Broadcasted{DefaultArrayStyle{1}, Nothing, typeof(*), Tuple{Int64, Broadcasted{DefaultArrayStyle{1}, Nothing, typeof(-), Tuple{Vector{Float64}, Vector{Float64}}}}}})   # time: 0.021802723
    Base.precompile(Tuple{typeof(broadcasted),typeof(*),Int64,Broadcasted{DefaultArrayStyle{1}, Nothing, typeof(-), Tuple{Vector{Float64}, Vector{Float64}}}})   # time: 0.004016119
    Base.precompile(Tuple{typeof(broadcasted),typeof(-),Vector{Float64},Vector{Float64}})   # time: 0.001601445
end
