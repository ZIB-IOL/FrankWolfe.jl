function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(broadcasted),typeof(-),Any,Vector{Float64}})   # time: 0.7300752
    Base.precompile(Tuple{typeof(preprocess_args),Nothing,Tuple{Any}})   # time: 0.050907776
    Base.precompile(Tuple{typeof(promote_typejoin_union),Type})   # time: 0.00595199
    Base.precompile(Tuple{typeof(broadcasted),Function,Vector{Float64},Vector{Float64}})   # time: 0.004084394
    Base.precompile(Tuple{typeof(broadcasted),Function,Float64,Vector{Float64}})   # time: 0.004078107
    Base.precompile(Tuple{typeof(broadcasted),Function,Vector{Float64},Broadcasted{DefaultArrayStyle{1}, Nothing, typeof(*), Tuple{Float64, Vector{Float64}}}})   # time: 0.004077269
end
