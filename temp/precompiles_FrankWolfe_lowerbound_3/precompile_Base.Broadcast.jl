function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(materialize!),DefaultArrayStyle{1},Vector{Float64},Broadcasted{DefaultArrayStyle{1}, Nothing, typeof(+), Tuple{Vector{Float64}, Broadcasted{DefaultArrayStyle{1}, Nothing, typeof(*), Tuple{Float64, Broadcasted{DefaultArrayStyle{0}, Nothing, typeof(-), Tuple{Int64, Float64}}, Vector{Float64}}}}}})   # time: 0.033894487
    Base.precompile(Tuple{typeof(broadcasted),Function,Float64,Broadcasted{DefaultArrayStyle{0}, Nothing, typeof(-), Tuple{Int64, Float64}},Vector{Float64}})   # time: 0.006022778
    Base.precompile(Tuple{typeof(getindex),Broadcasted{Nothing, Tuple{OneTo{Int64}}, typeof(*), Tuple{Int64, Broadcasted{DefaultArrayStyle{1}, Nothing, typeof(-), Tuple{Extruded{Vector{Float64}, Tuple{Bool}, Tuple{Int64}}, Extruded{Vector{Float64}, Tuple{Bool}, Tuple{Int64}}}}}},Int64})   # time: 0.003313103
    Base.precompile(Tuple{typeof(broadcasted),typeof(-),Int64,Float64})   # time: 0.002890142
    Base.precompile(Tuple{typeof(broadcasted),typeof(-),Int64,BigFloat})   # time: 0.002712121
    Base.precompile(Tuple{Type{Broadcasted{DefaultArrayStyle{1}}},typeof(*),Tuple{Float64, Broadcasted{DefaultArrayStyle{0}, Nothing, typeof(-), Tuple{Int64, Float64}}, Vector{Float64}}})   # time: 0.00235719
    Base.precompile(Tuple{Type{Broadcasted{DefaultArrayStyle{1}}},typeof(+),Tuple{Vector{Float64}, Broadcasted{DefaultArrayStyle{1}, Nothing, typeof(*), Tuple{Float64, Broadcasted{DefaultArrayStyle{0}, Nothing, typeof(-), Tuple{Int64, Float64}}, Vector{Float64}}}}})   # time: 0.001800214
end
