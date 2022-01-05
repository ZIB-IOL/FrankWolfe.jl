function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(broadcasted),typeof(-),Vector{Float64},AbstractVector})   # time: 0.080118395
    Base.precompile(Tuple{typeof(materialize!),Vector{Float64},Broadcasted{Style}})   # time: 0.004356678
    Base.precompile(Tuple{typeof(broadcasted),DefaultArrayStyle{1},Function,Vector{Float64},Vector{Float64}})   # time: 0.00343813
end
