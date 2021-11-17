function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(materialize!),Vector{Float64},Broadcasted{DefaultArrayStyle{1}, Nothing, typeof(-), Tuple{Vector{Float64}, Base.RefValue{Nothing}}}})   # time: 0.025456129
    Base.precompile(Tuple{typeof(broadcasted),typeof(-),Vector{Float64},Nothing})   # time: 0.006213332
    Base.precompile(Tuple{typeof(materialize!),Vector{Float64},Broadcasted{DefaultArrayStyle{1}, Nothing, typeof(-), Tuple{Vector{Float64}, Vector{Float64}}}})   # time: 0.005823833
end
