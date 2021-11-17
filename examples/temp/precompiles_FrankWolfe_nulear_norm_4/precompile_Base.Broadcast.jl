function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(materialize!),DefaultArrayStyle{1},Vector{Float64},Broadcasted{DefaultArrayStyle{1}, Nothing, typeof(+), Tuple{Vector{Float64}, Broadcasted{DefaultArrayStyle{1}, Nothing, typeof(*), Tuple{Float64, Broadcasted{DefaultArrayStyle{0}, Nothing, typeof(-), Tuple{Int64, Float64}}, Vector{Float64}}}}}})   # time: 0.04034316
    Base.precompile(Tuple{typeof(materialize!),DefaultArrayStyle{1},Vector{Float64},Broadcasted{DefaultArrayStyle{1}, Nothing, typeof(-), Tuple{Vector{Float64}, Broadcasted{DefaultArrayStyle{1}, Nothing, typeof(*), Tuple{Float64, Vector{Float64}}}}}})   # time: 0.024052173
    Base.precompile(Tuple{typeof(broadcasted),Function,Float64,Broadcasted{DefaultArrayStyle{0}, Nothing, typeof(-), Tuple{Int64, Float64}},Vector{Float64}})   # time: 0.006933977
    Base.precompile(Tuple{Type{Broadcasted{DefaultArrayStyle{1}, Axes, F, Args} where {Axes, F, Args<:Tuple}},typeof(+),Tuple{Vector{Float64}, Broadcasted{DefaultArrayStyle{1}, Nothing, typeof(*), Tuple{Float64, Broadcasted{DefaultArrayStyle{0}, Nothing, typeof(-), Tuple{Int64, Float64}}, Vector{Float64}}}}})   # time: 0.003918889
    Base.precompile(Tuple{Type{Broadcasted{DefaultArrayStyle{1}, Axes, F, Args} where {Axes, F, Args<:Tuple}},typeof(*),Tuple{Float64, Broadcasted{DefaultArrayStyle{0}, Nothing, typeof(-), Tuple{Int64, Float64}}, Vector{Float64}}})   # time: 0.002461877
    Base.precompile(Tuple{Type{Broadcasted{DefaultArrayStyle{1}, Axes, F, Args} where {Axes, F, Args<:Tuple}},typeof(-),Tuple{Vector{Float64}, Broadcasted{DefaultArrayStyle{1}, Nothing, typeof(*), Tuple{Float64, Vector{Float64}}}}})   # time: 0.002063228
end
