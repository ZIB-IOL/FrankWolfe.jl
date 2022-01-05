function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(materialize!),Vector{Float64},Broadcasted{DefaultArrayStyle{1}, Nothing, typeof(*), Tuple{Int64, Broadcasted{DefaultArrayStyle{1}, Nothing, typeof(-), Tuple{Vector{Float64}, Vector{Float64}}}}}})   # time: 0.021332275
    Base.precompile(Tuple{typeof(broadcasted),typeof(-),Vector{Float64},Any})   # time: 0.015509285
    Base.precompile(Tuple{typeof(materialize!),Vector{Float64},Broadcasted{Style}})   # time: 0.00960906
    Base.precompile(Tuple{typeof(broadcasted),Function,Int64,Broadcasted{DefaultArrayStyle{1}, Nothing, typeof(-), Tuple{Vector{Float64}, Vector{Float64}}}})   # time: 0.003694306
    Base.precompile(Tuple{typeof(broadcasted),Function,Vector{Float64},Vector{Float64}})   # time: 0.003401462
    Base.precompile(Tuple{Type{Broadcasted{DefaultArrayStyle{1}}},typeof(*),Tuple{Int64, Broadcasted{DefaultArrayStyle{1}, Nothing, typeof(-), Tuple{Vector{Float64}, Vector{Float64}}}}})   # time: 0.001127929
end
