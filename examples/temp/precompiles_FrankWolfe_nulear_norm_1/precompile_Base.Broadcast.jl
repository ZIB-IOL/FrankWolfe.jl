function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(materialize!),Matrix{Float64},Broadcasted{DefaultArrayStyle{2}, Nothing, typeof(-), Tuple{Matrix{Float64}, Broadcasted{DefaultArrayStyle{2}, Nothing, typeof(*), Tuple{Float64, Matrix{Float64}}}}}})   # time: 0.03476911
    Base.precompile(Tuple{typeof(materialize),Broadcasted{DefaultArrayStyle{2}, Nothing, typeof(big), Tuple{Matrix{Float64}}}})   # time: 0.032892786
    Base.precompile(Tuple{typeof(broadcasted),Function,Matrix{Float64},Broadcasted{DefaultArrayStyle{2}, Nothing, typeof(*), Tuple{Float64, Matrix{Float64}}}})   # time: 0.006914041
    Base.precompile(Tuple{typeof(broadcasted),Function,Float64,Matrix{Float64}})   # time: 0.005706852
    Base.precompile(Tuple{typeof(broadcasted),typeof(big),Matrix{Float64}})   # time: 0.003356098
    Base.precompile(Tuple{typeof(broadcasted),typeof(+),Int64,UnitRange{Int64}})   # time: 0.002550428
    Base.precompile(Tuple{Type{Broadcasted{DefaultArrayStyle{2}, Axes, F, Args} where {Axes, F, Args<:Tuple}},typeof(-),Tuple{Matrix{Float64}, Broadcasted{DefaultArrayStyle{2}, Nothing, typeof(*), Tuple{Float64, Matrix{Float64}}}}})   # time: 0.002053861
    Base.precompile(Tuple{typeof(_broadcast_getindex),Extruded{Matrix{Float64}, Tuple{Bool, Bool}, Tuple{Int64, Int64}},CartesianIndex{2}})   # time: 0.001894274
    Base.precompile(Tuple{typeof(broadcasted),typeof(identity),Int64})   # time: 0.00104488
end
