function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(materialize!),Vector{Float64},Broadcasted{DefaultArrayStyle{1}, Nothing, typeof(+), Tuple{Vector{Float64}, Broadcasted{DefaultArrayStyle{1}, Nothing, typeof(*), Tuple{Float64, Broadcasted{DefaultArrayStyle{1}, Nothing, typeof(-), Tuple{Vector{Float64}, Vector{Float64}}}}}}}})   # time: 0.123147674
    Base.precompile(Tuple{typeof(materialize!),Vector{Float64},Broadcasted{DefaultArrayStyle{1}, Nothing, typeof(-), Tuple{Vector{Float64}, Float64}}})   # time: 0.08127204
    Base.precompile(Tuple{typeof(materialize!),Vector{Float64},Broadcasted{DefaultArrayStyle{1}, Nothing, typeof(-)}})   # time: 0.06171759
    Base.precompile(Tuple{typeof(materialize!),Vector{Float64},Broadcasted{DefaultArrayStyle{1}, Nothing, typeof(+), Tuple{Vector{Float64}, Broadcasted{DefaultArrayStyle{1}, Nothing, typeof(*), Tuple{BigFloat, Broadcasted{DefaultArrayStyle{1}, Nothing, typeof(-), Tuple{Vector{Float64}, Vector{Float64}}}}}}}})   # time: 0.041820396
    Base.precompile(Tuple{typeof(materialize!),DefaultArrayStyle{1},Vector{Float64},Broadcasted{DefaultArrayStyle{1}, Nothing, typeof(+), Tuple{Vector{Float64}, Broadcasted{DefaultArrayStyle{1}, Nothing, typeof(*), Tuple{Float64, Broadcasted{DefaultArrayStyle{0}, Nothing, typeof(-), Tuple{Int64, Float64}}, Vector{Float64}}}}}})   # time: 0.03354683
    Base.precompile(Tuple{typeof(materialize!),Vector{Float64},Broadcasted{DefaultArrayStyle{1}, Nothing, typeof(+)}})   # time: 0.029576246
    Base.precompile(Tuple{typeof(materialize!),Vector{BigFloat},Broadcasted{DefaultArrayStyle{1}, Nothing, typeof(-), Tuple{Vector{BigFloat}, BigFloat}}})   # time: 0.027423015
    Base.precompile(Tuple{typeof(broadcasted),typeof(*),Union{Float64, BigFloat},Union{Broadcasted{DefaultArrayStyle{0}, Nothing, typeof(-), Tuple{Int64, Float64}}, Broadcasted{DefaultArrayStyle{0}, Nothing, typeof(-), Tuple{Int64, BigFloat}}},Union{Vector{Float64}, Vector{BigFloat}}})   # time: 0.010900958
    Base.precompile(Tuple{typeof(broadcasted),typeof(-),Vector{Float64},Broadcasted{DefaultArrayStyle{1}, Nothing, typeof(*)}})   # time: 0.010339007
    Base.precompile(Tuple{typeof(broadcasted),typeof(+),Vector{Float64},Broadcasted{DefaultArrayStyle{1}, Nothing, typeof(*)}})   # time: 0.00705384
    Base.precompile(Tuple{typeof(broadcasted),typeof(-),Vector{BigFloat},BigFloat})   # time: 0.006022767
    Base.precompile(Tuple{typeof(broadcasted),typeof(*),BigFloat,Vector{Float64}})   # time: 0.005838032
    Base.precompile(Tuple{typeof(broadcasted),typeof(+),Vector{Float64},Broadcasted{DefaultArrayStyle{1}, Nothing, typeof(*), Tuple{BigFloat, Broadcasted{DefaultArrayStyle{1}, Nothing, typeof(-), Tuple{Vector{Float64}, Vector{Float64}}}}}})   # time: 0.005750656
    Base.precompile(Tuple{typeof(broadcasted),Function,Float64,Broadcasted{DefaultArrayStyle{0}, Nothing, typeof(-), Tuple{Int64, Float64}},Vector{Float64}})   # time: 0.005655325
    Base.precompile(Tuple{typeof(broadcasted),typeof(+),Vector{Float64},Broadcasted{DefaultArrayStyle{1}, Nothing, typeof(*), Tuple{Float64, Broadcasted{DefaultArrayStyle{1}, Nothing, typeof(-), Tuple{Vector{Float64}, Vector{Float64}}}}}})   # time: 0.005511308
    Base.precompile(Tuple{typeof(broadcasted),typeof(-),Int64,BigFloat})   # time: 0.005470661
    Base.precompile(Tuple{typeof(broadcasted),typeof(*),Float64,Broadcasted{DefaultArrayStyle{1}, Nothing, typeof(-), Tuple{Vector{Float64}, Vector{Float64}}}})   # time: 0.00525834
    Base.precompile(Tuple{typeof(broadcasted),typeof(*),Float64,Vector{BigFloat}})   # time: 0.003710429
    Base.precompile(Tuple{typeof(broadcasted),typeof(*),BigFloat,Broadcasted{DefaultArrayStyle{1}, Nothing, typeof(-), Tuple{Vector{Float64}, Vector{Float64}}}})   # time: 0.003657909
    Base.precompile(Tuple{typeof(broadcasted),typeof(*),BigFloat,Vector{BigFloat}})   # time: 0.003648761
    Base.precompile(Tuple{typeof(broadcasted),typeof(-),Int64,Float64})   # time: 0.00330032
    Base.precompile(Tuple{typeof(broadcasted),typeof(identity),Int64})   # time: 0.003120125
    Base.precompile(Tuple{typeof(broadcasted),typeof(-),Vector{Float64},Float64})   # time: 0.002741443
    Base.precompile(Tuple{Type{Broadcasted{DefaultArrayStyle{1}}},typeof(+),Tuple{Vector{Float64}, Broadcasted{DefaultArrayStyle{1}, Nothing, typeof(*), Tuple{Float64, Broadcasted{DefaultArrayStyle{0}, Nothing, typeof(-), Tuple{Int64, Float64}}, Vector{Float64}}}}})   # time: 0.002107136
    Base.precompile(Tuple{Type{Broadcasted{DefaultArrayStyle{1}}},typeof(*),Tuple{Float64, Broadcasted{DefaultArrayStyle{0}, Nothing, typeof(-), Tuple{Int64, Float64}}, Vector{Float64}}})   # time: 0.0018974
end
