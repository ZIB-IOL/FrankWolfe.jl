function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(materialize!),Vector{Float64},Broadcasted{DefaultArrayStyle{1}, Nothing, typeof(-), _A} where _A<:Tuple})   # time: 0.042412344
    Base.precompile(Tuple{typeof(materialize!),DefaultArrayStyle{1},Vector{Float64},Broadcasted{DefaultArrayStyle{1}, Nothing, typeof(+), Tuple{Vector{Float64}, Broadcasted{DefaultArrayStyle{1}, Nothing, typeof(*), Tuple{Float64, Broadcasted{DefaultArrayStyle{0}, Nothing, typeof(-), Tuple{Int64, Float64}}, Vector{Float64}}}}}})   # time: 0.0416976
    Base.precompile(Tuple{typeof(materialize!),Vector{Float64},Broadcasted{DefaultArrayStyle{1}, Nothing, typeof(+), _A} where _A<:Tuple})   # time: 0.03072932
    Base.precompile(Tuple{typeof(materialize!),DefaultArrayStyle{1},Vector{Float64},Broadcasted{DefaultArrayStyle{1}, Nothing, typeof(-), Tuple{Vector{Float64}, Broadcasted{DefaultArrayStyle{1}, Nothing, typeof(*), Tuple{Float64, Vector{Float64}}}}}})   # time: 0.021879753
    Base.precompile(Tuple{typeof(materialize!),Vector{BigFloat},Broadcasted{DefaultArrayStyle{1}, Nothing, typeof(-), Tuple{Vector{BigFloat}, BigFloat}}})   # time: 0.017311452
    Base.precompile(Tuple{typeof(materialize!),Vector{Float64},Broadcasted{DefaultArrayStyle{1}, Nothing, typeof(-), Tuple{Vector{Float64}, Float64}}})   # time: 0.0161136
    Base.precompile(Tuple{typeof(broadcasted),Function,Float64,Broadcasted{DefaultArrayStyle{0}, Nothing, typeof(-), Tuple{Int64, Float64}},Vector{Float64}})   # time: 0.007179249
    Base.precompile(Tuple{typeof(broadcasted),typeof(*),Union{Float64, BigFloat},Union{Broadcasted{DefaultArrayStyle{0}, Nothing, typeof(-), Tuple{Int64, Float64}}, Broadcasted{DefaultArrayStyle{0}, Nothing, typeof(-), Tuple{Int64, BigFloat}}},Union{Vector{Float64}, Vector{BigFloat}}})   # time: 0.006824313
    Base.precompile(Tuple{typeof(broadcasted),typeof(-),Vector{Float64},Broadcasted{DefaultArrayStyle{1}, Nothing, typeof(*), Args} where Args<:Tuple})   # time: 0.006067518
    Base.precompile(Tuple{typeof(broadcasted),typeof(-),Vector{Float64},Float64})   # time: 0.003987787
    Base.precompile(Tuple{typeof(broadcasted),typeof(+),Vector{Float64},Broadcasted{DefaultArrayStyle{1}, Nothing, typeof(*), _A} where _A<:Tuple})   # time: 0.0039653
    Base.precompile(Tuple{Type{Broadcasted{DefaultArrayStyle{1}, Axes, F, Args} where {Axes, F, Args<:Tuple}},typeof(*),Tuple{Float64, Broadcasted{DefaultArrayStyle{0}, Nothing, typeof(-), Tuple{Int64, Float64}}, Vector{Float64}}})   # time: 0.002793714
    Base.precompile(Tuple{typeof(broadcasted),typeof(-),Int64,Float64})   # time: 0.00267268
    Base.precompile(Tuple{Type{Broadcasted{DefaultArrayStyle{1}, Axes, F, Args} where {Axes, F, Args<:Tuple}},typeof(+),Tuple{Vector{Float64}, Broadcasted{DefaultArrayStyle{1}, Nothing, typeof(*), Tuple{Float64, Broadcasted{DefaultArrayStyle{0}, Nothing, typeof(-), Tuple{Int64, Float64}}, Vector{Float64}}}}})   # time: 0.002649004
    Base.precompile(Tuple{typeof(broadcasted),typeof(-),Int64,BigFloat})   # time: 0.00264125
    Base.precompile(Tuple{typeof(broadcasted),typeof(-),Vector{BigFloat},BigFloat})   # time: 0.002600257
    Base.precompile(Tuple{Type{Broadcasted{DefaultArrayStyle{1}, Axes, F, Args} where {Axes, F, Args<:Tuple}},typeof(-),Tuple{Vector{Float64}, Broadcasted{DefaultArrayStyle{1}, Nothing, typeof(*), Tuple{Float64, Vector{Float64}}}}})   # time: 0.002324383
    Base.precompile(Tuple{typeof(broadcasted),typeof(identity),Int64})   # time: 0.001067312
end
