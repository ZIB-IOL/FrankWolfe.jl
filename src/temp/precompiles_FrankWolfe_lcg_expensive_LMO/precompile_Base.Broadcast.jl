function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(broadcasted),typeof(Base.literal_pow),typeof(^),Any,Val{2}})   # time: 0.7835579
    isdefined(Base.Broadcast, Symbol("#5#6")) && Base.precompile(Tuple{getfield(Base.Broadcast, Symbol("#5#6")),Int64,Float64,Vararg{Any, N} where N})   # time: 0.002957702
    isdefined(Base.Broadcast, Symbol("#8#10")) && Base.precompile(Tuple{getfield(Base.Broadcast, Symbol("#8#10")),Function,Int64,Val{2}})   # time: 0.001609207
    isdefined(Base.Broadcast, Symbol("#8#10")) && Base.precompile(Tuple{getfield(Base.Broadcast, Symbol("#8#10")),Float64,Float64,Function,Int64,Val{2}})   # time: 0.00152149
end
