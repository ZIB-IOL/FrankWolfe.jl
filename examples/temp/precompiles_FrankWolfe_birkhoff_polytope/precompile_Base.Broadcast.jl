function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(materialize!),Vector{Float64},Broadcasted{DefaultArrayStyle{1}, Nothing, typeof(-), _A} where _A<:Tuple})   # time: 0.054268297
    Base.precompile(Tuple{typeof(broadcasted),typeof(*),Union{Float64, BigFloat},Union{Broadcasted{DefaultArrayStyle{0}, Nothing, typeof(-), Tuple{Int64, Float64}}, Broadcasted{DefaultArrayStyle{0}, Nothing, typeof(-), Tuple{Int64, BigFloat}}},Union{Vector{Float64}, Vector{BigFloat}}})   # time: 0.03626169
    Base.precompile(Tuple{typeof(broadcasted),typeof(*),Float64,Vector{Float64}})   # time: 0.029384669
    Base.precompile(Tuple{typeof(materialize!),DefaultArrayStyle{1},Vector{Float64},Broadcasted{DefaultArrayStyle{1}, Nothing, typeof(+), Tuple{Vector{Float64}, Broadcasted{DefaultArrayStyle{1}, Nothing, typeof(*), Tuple{Float64, Broadcasted{DefaultArrayStyle{0}, Nothing, typeof(-), Tuple{Int64, Float64}}, Vector{Float64}}}}}})   # time: 0.021901097
    Base.precompile(Tuple{typeof(materialize!),Vector{Float64},Broadcasted{DefaultArrayStyle{1}, Nothing, typeof(+), _A} where _A<:Tuple})   # time: 0.019222312
    Base.precompile(Tuple{typeof(materialize!),DefaultArrayStyle{1},Vector{Float64},Broadcasted{DefaultArrayStyle{1}, Nothing, typeof(-), Tuple{Vector{Float64}, Broadcasted{DefaultArrayStyle{1}, Nothing, typeof(*), Tuple{Float64, Vector{Float64}}}}}})   # time: 0.015836336
    Base.precompile(Tuple{typeof(materialize!),Vector{BigFloat},Broadcasted{DefaultArrayStyle{1}, Nothing, typeof(-), Tuple{Vector{BigFloat}, BigFloat}}})   # time: 0.013281195
    Base.precompile(Tuple{typeof(materialize!),Vector{Float64},Broadcasted{DefaultArrayStyle{1}, Nothing, typeof(-), Tuple{Vector{Float64}, Float64}}})   # time: 0.011597763
    Base.precompile(Tuple{typeof(broadcasted),Function,Float64,Broadcasted{DefaultArrayStyle{0}, Nothing, typeof(-), Tuple{Int64, Float64}},Vector{Float64}})   # time: 0.004279247
    Base.precompile(Tuple{typeof(broadcasted),typeof(-),Vector{Float64},Broadcasted{DefaultArrayStyle{1}, Nothing, typeof(*), Args} where Args<:Tuple})   # time: 0.003729356
    Base.precompile(Tuple{typeof(broadcasted),typeof(+),Vector{Float64},Broadcasted{DefaultArrayStyle{1}, Nothing, typeof(*), _A} where _A<:Tuple})   # time: 0.002882748
    Base.precompile(Tuple{typeof(broadcasted),typeof(-),Vector{Float64},Float64})   # time: 0.00246245
    Base.precompile(Tuple{typeof(broadcasted),typeof(*),BigFloat,Vector{Float64}})   # time: 0.002325304
    isdefined(Base.Broadcast, Symbol("#5#6")) && Base.precompile(Tuple{getfield(Base.Broadcast, Symbol("#5#6")),Float64,Int64,Vararg{Any, N} where N})   # time: 0.001980275
    Base.precompile(Tuple{typeof(broadcasted),typeof(-),Vector{BigFloat},BigFloat})   # time: 0.001840064
    Base.precompile(Tuple{typeof(broadcasted),typeof(-),Int64,BigFloat})   # time: 0.001772395
    Base.precompile(Tuple{Type{Broadcasted{DefaultArrayStyle{1}, Axes, F, Args} where {Axes, F, Args<:Tuple}},typeof(-),Tuple{Vector{Float64}, Broadcasted{DefaultArrayStyle{1}, Nothing, typeof(*), Tuple{Float64, Vector{Float64}}}}})   # time: 0.001430234
    Base.precompile(Tuple{Type{Broadcasted{DefaultArrayStyle{1}, Axes, F, Args} where {Axes, F, Args<:Tuple}},typeof(+),Tuple{Vector{Float64}, Broadcasted{DefaultArrayStyle{1}, Nothing, typeof(*), Tuple{Float64, Broadcasted{DefaultArrayStyle{0}, Nothing, typeof(-), Tuple{Int64, Float64}}, Vector{Float64}}}}})   # time: 0.001364871
    isdefined(Base.Broadcast, Symbol("#5#6")) && Base.precompile(Tuple{getfield(Base.Broadcast, Symbol("#5#6")),Int64,Float64,Vararg{Any, N} where N})   # time: 0.001315856
    Base.precompile(Tuple{typeof(broadcasted),typeof(*),Float64,Vector{BigFloat}})   # time: 0.001296723
    Base.precompile(Tuple{Type{Broadcasted{DefaultArrayStyle{1}, Axes, F, Args} where {Axes, F, Args<:Tuple}},typeof(*),Tuple{Float64, Broadcasted{DefaultArrayStyle{0}, Nothing, typeof(-), Tuple{Int64, Float64}}, Vector{Float64}}})   # time: 0.001281496
    Base.precompile(Tuple{typeof(broadcasted),typeof(*),BigFloat,Vector{BigFloat}})   # time: 0.001195959
    isdefined(Base.Broadcast, Symbol("#8#10")) && Base.precompile(Tuple{getfield(Base.Broadcast, Symbol("#8#10")),Function,Int64,Val{2}})   # time: 0.001043731
    isdefined(Base.Broadcast, Symbol("#8#10")) && Base.precompile(Tuple{getfield(Base.Broadcast, Symbol("#8#10")),Float64,Float64,Function,Int64,Val{2}})   # time: 0.001004558
end
