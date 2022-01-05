function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(materialize!),Vector{Float64},Broadcasted{DefaultArrayStyle{1}, Nothing, typeof(+), Tuple{Broadcasted{DefaultArrayStyle{1}, Nothing, typeof(*), Tuple{Broadcasted{DefaultArrayStyle{0}, Nothing, typeof(-), Tuple{Int64, Float64}}, Vector{Float64}}}, Broadcasted{DefaultArrayStyle{1}, Nothing, typeof(*), Tuple{Float64, Vector{Float64}}}}}})   # time: 0.062369872
    Base.precompile(Tuple{typeof(broadcasted),typeof(*),Broadcasted{DefaultArrayStyle{0}, Nothing, typeof(-), Tuple{Int64, Float64}},Vector{Float64}})   # time: 0.007064855
    Base.precompile(Tuple{typeof(broadcasted),typeof(-),Int64,Float64})   # time: 0.005906828
    Base.precompile(Tuple{typeof(broadcasted),typeof(+),Broadcasted{DefaultArrayStyle{1}, Nothing, typeof(*), Tuple{Broadcasted{DefaultArrayStyle{0}, Nothing, typeof(-), Tuple{Int64, Float64}}, Vector{Float64}}},Broadcasted{DefaultArrayStyle{1}, Nothing, typeof(*), Tuple{Float64, Vector{Float64}}}})   # time: 0.005412011
    Base.precompile(Tuple{typeof(materialize!),Vector{Float64},Broadcasted{DefaultArrayStyle{0}, Nothing, typeof(identity), Tuple{Int64}}})   # time: 0.004651796
    Base.precompile(Tuple{typeof(broadcasted),typeof(identity),Int64})   # time: 0.00306013
end
