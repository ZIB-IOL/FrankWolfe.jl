function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(materialize!),Vector{Float64},Broadcasted{DefaultArrayStyle{1}, Nothing, typeof(*), Tuple{Int64, Broadcasted{DefaultArrayStyle{1}, Nothing, typeof(-), Tuple{Vector{Float64}, Vector{Float64}}}}}})   # time: 0.047182348
    Base.precompile(Tuple{typeof(materialize!),Vector{Float64},Broadcasted{DefaultArrayStyle{1}, Nothing, typeof(-), Tuple{Vector{Float64}, Broadcasted{DefaultArrayStyle{1}, Nothing, typeof(*), Tuple{Float64, Vector{Float64}}}}}})   # time: 0.035110768
    Base.precompile(Tuple{typeof(broadcasted),typeof(*),Int64,Broadcasted{DefaultArrayStyle{1}, Nothing, typeof(-), Tuple{Vector{Float64}, Vector{Float64}}}})   # time: 0.006603323
    Base.precompile(Tuple{typeof(broadcasted),typeof(-),Vector{Float64},Broadcasted{DefaultArrayStyle{1}, Nothing, typeof(*), Tuple{Float64, Vector{Float64}}}})   # time: 0.004778121
    Base.precompile(Tuple{typeof(broadcasted),typeof(-),Vector{Float64},Vector{Float64}})   # time: 0.004534165
    Base.precompile(Tuple{typeof(broadcasted),typeof(*),Float64,Vector{Float64}})   # time: 0.002344816
end
