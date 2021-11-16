function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(materialize!),Matrix{Float64},Broadcasted{DefaultArrayStyle{2}, Nothing, typeof(-), Tuple{Matrix{Float64}, Broadcasted{DefaultArrayStyle{2}, Nothing, typeof(*), Tuple{Float64, Matrix{Float64}}}}}})   # time: 0.032148853
    Base.precompile(Tuple{typeof(materialize!),Matrix{Float64},Broadcasted{DefaultArrayStyle{2}, Nothing, typeof(-), _A} where _A<:Tuple})   # time: 0.024126839
    Base.precompile(Tuple{typeof(broadcasted),typeof(-),Matrix{Float64},Any})   # time: 0.019169925
    Base.precompile(Tuple{typeof(materialize),Broadcasted{DefaultArrayStyle{2}, Nothing, typeof(big), Tuple{Matrix{Float64}}}})   # time: 0.016311536
    Base.precompile(Tuple{typeof(materialize!),Matrix{Float64},Broadcasted{Style, Axes, F, Args} where {Axes, F, Args<:Tuple}})   # time: 0.010250095
    Base.precompile(Tuple{typeof(broadcasted),Function,Float64,Matrix{Float64}})   # time: 0.006460444
    Base.precompile(Tuple{typeof(broadcasted),Function,Matrix{Float64},Broadcasted{DefaultArrayStyle{2}, Nothing, typeof(*), Tuple{Float64, Matrix{Float64}}}})   # time: 0.005801843
    Base.precompile(Tuple{typeof(broadcasted),typeof(big),Matrix{Float64}})   # time: 0.002693285
    Base.precompile(Tuple{Type{Broadcasted{DefaultArrayStyle{2}, Axes, F, Args} where {Axes, F, Args<:Tuple}},typeof(-),Tuple{Matrix{Float64}, Broadcasted{DefaultArrayStyle{2}, Nothing, typeof(*), Tuple{Float64, Matrix{Float64}}}}})   # time: 0.002513653
    Base.precompile(Tuple{typeof(broadcasted),typeof(identity),Int64})   # time: 0.001802953
    Base.precompile(Tuple{typeof(_broadcast_getindex),Extruded{Matrix{Float64}, Tuple{Bool, Bool}, Tuple{Int64, Int64}},CartesianIndex{2}})   # time: 0.001239125
end
