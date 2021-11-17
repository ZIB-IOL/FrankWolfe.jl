function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(materialize!),Matrix{Float64},Broadcasted{DefaultArrayStyle{0}, Nothing, typeof(identity), Tuple{Int64}}})   # time: 0.003203544
    Base.precompile(Tuple{typeof(broadcasted),typeof(identity),Int64})   # time: 0.001142672
end
