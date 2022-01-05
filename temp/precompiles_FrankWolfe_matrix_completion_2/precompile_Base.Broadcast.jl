function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(broadcasted),typeof(-),Matrix{Float64},Any})   # time: 0.017525315
    Base.precompile(Tuple{typeof(materialize!),Matrix{Float64},Broadcasted{Style}})   # time: 0.0117163
end
