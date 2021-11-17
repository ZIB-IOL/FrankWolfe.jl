function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(materialize!),DefaultArrayStyle{1},Vector{Float64},Broadcasted{DefaultArrayStyle{1}, Nothing, typeof(+), Tuple{Vector{Float64}, Broadcasted{DefaultArrayStyle{1}, Nothing, typeof(*), Tuple{Float64, Broadcasted{DefaultArrayStyle{0}, Nothing, typeof(-), Tuple{Int64, Float64}}, Vector{Float64}}}}}})   # time: 0.25637764
    Base.precompile(Tuple{typeof(materialize!),DefaultArrayStyle{1},Vector{Float64},Broadcasted{DefaultArrayStyle{1}, Nothing, typeof(-), Tuple{Vector{Float64}, Broadcasted{DefaultArrayStyle{1}, Nothing, typeof(*), Tuple{Float64, Vector{Float64}}}}}})   # time: 0.02666254
    Base.precompile(Tuple{typeof(broadcasted),Function,Float64,Broadcasted{DefaultArrayStyle{0}, Nothing, typeof(-), Tuple{Int64, Float64}},Vector{Float64}})   # time: 0.005984048
    Base.precompile(Tuple{Type{Broadcasted{DefaultArrayStyle{1}, Axes, F, Args} where {Axes, F, Args<:Tuple}},typeof(+),Tuple{Vector{Float64}, Broadcasted{DefaultArrayStyle{1}, Nothing, typeof(*), Tuple{Float64, Broadcasted{DefaultArrayStyle{0}, Nothing, typeof(-), Tuple{Int64, Float64}}, Vector{Float64}}}}})   # time: 0.001983114
    Base.precompile(Tuple{Type{Broadcasted{DefaultArrayStyle{1}, Axes, F, Args} where {Axes, F, Args<:Tuple}},typeof(-),Tuple{Vector{Float64}, Broadcasted{DefaultArrayStyle{1}, Nothing, typeof(*), Tuple{Float64, Vector{Float64}}}}})   # time: 0.001938273
    Base.precompile(Tuple{Type{Broadcasted{DefaultArrayStyle{1}, Axes, F, Args} where {Axes, F, Args<:Tuple}},typeof(*),Tuple{Float64, Broadcasted{DefaultArrayStyle{0}, Nothing, typeof(-), Tuple{Int64, Float64}}, Vector{Float64}}})   # time: 0.001884918
end
