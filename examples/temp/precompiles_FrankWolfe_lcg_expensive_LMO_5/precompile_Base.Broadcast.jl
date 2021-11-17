function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(materialize!),Vector{Float64},Broadcasted{DefaultArrayStyle{1}, Nothing, typeof(-), _A} where _A<:Tuple})   # time: 0.099250756
    Base.precompile(Tuple{typeof(materialize!),DefaultArrayStyle{1},Vector{Float64},Broadcasted{DefaultArrayStyle{1}, Nothing, typeof(+), Tuple{Vector{Float64}, Broadcasted{DefaultArrayStyle{1}, Nothing, typeof(*), Tuple{Float64, Broadcasted{DefaultArrayStyle{0}, Nothing, typeof(-), Tuple{Int64, Float64}}, Vector{Float64}}}}}})   # time: 0.037033364
    Base.precompile(Tuple{typeof(materialize!),Vector{Float64},Broadcasted{DefaultArrayStyle{1}, Nothing, typeof(+), _A} where _A<:Tuple})   # time: 0.03573895
    Base.precompile(Tuple{typeof(materialize!),DefaultArrayStyle{1},Vector{Float64},Broadcasted{DefaultArrayStyle{1}, Nothing, typeof(-), Tuple{Vector{Float64}, Broadcasted{DefaultArrayStyle{1}, Nothing, typeof(*), Tuple{Float64, Vector{Float64}}}}}})   # time: 0.022698855
    Base.precompile(Tuple{typeof(materialize!),Vector{BigFloat},Broadcasted{DefaultArrayStyle{1}, Nothing, typeof(-), Tuple{Vector{BigFloat}, BigFloat}}})   # time: 0.0218577
    Base.precompile(Tuple{typeof(materialize!),Vector{Float64},Broadcasted{DefaultArrayStyle{1}, Nothing, typeof(-), Tuple{Vector{Float64}, Float64}}})   # time: 0.020449154
    Base.precompile(Tuple{typeof(broadcasted),typeof(*),Union{Float64, BigFloat},Union{Broadcasted{DefaultArrayStyle{0}, Nothing, typeof(-), Tuple{Int64, Float64}}, Broadcasted{DefaultArrayStyle{0}, Nothing, typeof(-), Tuple{Int64, BigFloat}}},Union{Vector{Float64}, Vector{BigFloat}}})   # time: 0.00980613
    Base.precompile(Tuple{typeof(broadcasted),typeof(-),Vector{Float64},Broadcasted{DefaultArrayStyle{1}, Nothing, typeof(*), Args} where Args<:Tuple})   # time: 0.008262035
    Base.precompile(Tuple{typeof(broadcasted),Function,Float64,Broadcasted{DefaultArrayStyle{0}, Nothing, typeof(-), Tuple{Int64, Float64}},Vector{Float64}})   # time: 0.006576859
    Base.precompile(Tuple{typeof(broadcasted),typeof(+),Vector{Float64},Broadcasted{DefaultArrayStyle{1}, Nothing, typeof(*), _A} where _A<:Tuple})   # time: 0.006284644
    Base.precompile(Tuple{typeof(broadcasted),typeof(-),Vector{Float64},Float64})   # time: 0.004590179
    Base.precompile(Tuple{typeof(broadcasted),typeof(-),Vector{BigFloat},BigFloat})   # time: 0.003222429
    Base.precompile(Tuple{Type{Broadcasted{DefaultArrayStyle{1}, Axes, F, Args} where {Axes, F, Args<:Tuple}},typeof(-),Tuple{Vector{Float64}, Broadcasted{DefaultArrayStyle{1}, Nothing, typeof(*), Tuple{Float64, Vector{Float64}}}}})   # time: 0.002441616
    Base.precompile(Tuple{Type{Broadcasted{DefaultArrayStyle{1}, Axes, F, Args} where {Axes, F, Args<:Tuple}},typeof(+),Tuple{Vector{Float64}, Broadcasted{DefaultArrayStyle{1}, Nothing, typeof(*), Tuple{Float64, Broadcasted{DefaultArrayStyle{0}, Nothing, typeof(-), Tuple{Int64, Float64}}, Vector{Float64}}}}})   # time: 0.002073138
    Base.precompile(Tuple{Type{Broadcasted{DefaultArrayStyle{1}, Axes, F, Args} where {Axes, F, Args<:Tuple}},typeof(*),Tuple{Float64, Broadcasted{DefaultArrayStyle{0}, Nothing, typeof(-), Tuple{Int64, Float64}}, Vector{Float64}}})   # time: 0.001950428
end
