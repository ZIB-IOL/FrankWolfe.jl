function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(materialize!),Vector{Float64},Broadcasted{DefaultArrayStyle{0}, Nothing, typeof(identity), Tuple{Int64}}})   # time: 0.004292897
    isdefined(Base.Broadcast, Symbol("#5#6")) && Base.precompile(Tuple{getfield(Base.Broadcast, Symbol("#5#6")),Float64,Int64,Vararg{Any, N} where N})   # time: 0.003446499
    isdefined(Base.Broadcast, Symbol("#8#10")) && Base.precompile(Tuple{getfield(Base.Broadcast, Symbol("#8#10")),Int64,Float64,Float64,Float64})   # time: 0.001665768
    Base.precompile(Tuple{typeof(broadcasted),typeof(identity),Int64})   # time: 0.001538241
    isdefined(Base.Broadcast, Symbol("#8#10")) && Base.precompile(Tuple{getfield(Base.Broadcast, Symbol("#8#10")),Float64,Float64})   # time: 0.001340248
end
