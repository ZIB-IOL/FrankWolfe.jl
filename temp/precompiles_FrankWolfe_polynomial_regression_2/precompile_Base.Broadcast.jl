function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(materialize!),Vector{Float64},Broadcasted{DefaultArrayStyle{1}, Nothing, typeof(-)}})   # time: 0.025336247
    Base.precompile(Tuple{typeof(materialize!),Vector{Float64},Broadcasted{DefaultArrayStyle{1}, Nothing, typeof(+), Tuple{Vector{Float64}, Broadcasted{DefaultArrayStyle{1}, Nothing, typeof(*), Tuple{Float64, Broadcasted{DefaultArrayStyle{1}, Nothing, typeof(-), Tuple{Vector{Float64}, Vector{Float64}}}}}}}})   # time: 0.021495845
    Base.precompile(Tuple{typeof(materialize!),DefaultArrayStyle{1},Vector{Float64},Broadcasted{DefaultArrayStyle{1}, Nothing, typeof(+), Tuple{Vector{Float64}, Broadcasted{DefaultArrayStyle{1}, Nothing, typeof(*), Tuple{Float64, Broadcasted{DefaultArrayStyle{0}, Nothing, typeof(-), Tuple{Int64, Float64}}, Vector{Float64}}}}}})   # time: 0.019299338
    Base.precompile(Tuple{typeof(materialize!),Vector{Float64},Broadcasted{DefaultArrayStyle{1}, Nothing, typeof(+), Tuple{Vector{Float64}, Broadcasted{DefaultArrayStyle{1}, Nothing, typeof(*), Tuple{BigFloat, Broadcasted{DefaultArrayStyle{1}, Nothing, typeof(-), Tuple{Vector{Float64}, Vector{Float64}}}}}}}})   # time: 0.018805627
    Base.precompile(Tuple{typeof(materialize!),Vector{BigFloat},Broadcasted{DefaultArrayStyle{1}, Nothing, typeof(-), Tuple{Vector{BigFloat}, BigFloat}}})   # time: 0.014718554
    Base.precompile(Tuple{typeof(materialize!),Vector{Float64},Broadcasted{DefaultArrayStyle{1}, Nothing, typeof(+)}})   # time: 0.013033842
    Base.precompile(Tuple{typeof(materialize!),Vector{Float64},Broadcasted{DefaultArrayStyle{1}, Nothing, typeof(-), Tuple{Vector{Float64}, Float64}}})   # time: 0.010019929
    Base.precompile(Tuple{typeof(broadcasted),typeof(*),Union{Float64, BigFloat},Union{Broadcasted{DefaultArrayStyle{0}, Nothing, typeof(-), Tuple{Int64, Float64}}, Broadcasted{DefaultArrayStyle{0}, Nothing, typeof(-), Tuple{Int64, BigFloat}}},Union{Vector{Float64}, Vector{BigFloat}}})   # time: 0.00564663
    Base.precompile(Tuple{typeof(broadcasted),typeof(-),Vector{Float64},Broadcasted{DefaultArrayStyle{1}, Nothing, typeof(*)}})   # time: 0.003867091
    Base.precompile(Tuple{typeof(broadcasted),Function,Float64,Broadcasted{DefaultArrayStyle{0}, Nothing, typeof(-), Tuple{Int64, Float64}},Vector{Float64}})   # time: 0.003612029
    Base.precompile(Tuple{typeof(broadcasted),typeof(+),Vector{Float64},Broadcasted{DefaultArrayStyle{1}, Nothing, typeof(*)}})   # time: 0.003025229
    Base.precompile(Tuple{typeof(broadcasted),typeof(-),Vector{BigFloat},BigFloat})   # time: 0.002842106
    Base.precompile(Tuple{typeof(broadcasted),typeof(*),Float64,Broadcasted{DefaultArrayStyle{1}, Nothing, typeof(-), Tuple{Vector{Float64}, Vector{Float64}}}})   # time: 0.00275124
    Base.precompile(Tuple{typeof(broadcasted),typeof(*),BigFloat,Vector{Float64}})   # time: 0.002469293
    Base.precompile(Tuple{typeof(broadcasted),typeof(+),Vector{Float64},Broadcasted{DefaultArrayStyle{1}, Nothing, typeof(*), Tuple{BigFloat, Broadcasted{DefaultArrayStyle{1}, Nothing, typeof(-), Tuple{Vector{Float64}, Vector{Float64}}}}}})   # time: 0.002389256
    Base.precompile(Tuple{typeof(broadcasted),typeof(-),Int64,BigFloat})   # time: 0.002384578
    Base.precompile(Tuple{typeof(broadcasted),typeof(+),Vector{Float64},Broadcasted{DefaultArrayStyle{1}, Nothing, typeof(*), Tuple{Float64, Broadcasted{DefaultArrayStyle{1}, Nothing, typeof(-), Tuple{Vector{Float64}, Vector{Float64}}}}}})   # time: 0.002359227
    Base.precompile(Tuple{typeof(broadcasted),typeof(*),BigFloat,Broadcasted{DefaultArrayStyle{1}, Nothing, typeof(-), Tuple{Vector{Float64}, Vector{Float64}}}})   # time: 0.002184624
    Base.precompile(Tuple{typeof(broadcasted),typeof(*),Float64,Vector{BigFloat}})   # time: 0.002115338
    Base.precompile(Tuple{typeof(broadcasted),typeof(*),BigFloat,Vector{BigFloat}})   # time: 0.002032231
    Base.precompile(Tuple{typeof(broadcasted),typeof(-),Int64,Float64})   # time: 0.001590837
    Base.precompile(Tuple{typeof(broadcasted),typeof(-),Vector{Float64},Float64})   # time: 0.001520226
    Base.precompile(Tuple{Type{Broadcasted{DefaultArrayStyle{1}}},typeof(+),Tuple{Vector{Float64}, Broadcasted{DefaultArrayStyle{1}, Nothing, typeof(*), Tuple{Float64, Broadcasted{DefaultArrayStyle{0}, Nothing, typeof(-), Tuple{Int64, Float64}}, Vector{Float64}}}}})   # time: 0.001284795
    Base.precompile(Tuple{Type{Broadcasted{DefaultArrayStyle{1}}},typeof(*),Tuple{Float64, Broadcasted{DefaultArrayStyle{0}, Nothing, typeof(-), Tuple{Int64, Float64}}, Vector{Float64}}})   # time: 0.001259231
end
