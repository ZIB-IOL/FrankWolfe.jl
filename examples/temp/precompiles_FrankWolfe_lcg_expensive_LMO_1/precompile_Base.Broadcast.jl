function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    isdefined(Base.Broadcast, Symbol("#5#6")) && Base.precompile(Tuple{getfield(Base.Broadcast, Symbol("#5#6")),Int64,Float64,Vararg{Any, N} where N})   # time: 0.00149336
    isdefined(Base.Broadcast, Symbol("#8#10")) && Base.precompile(Tuple{getfield(Base.Broadcast, Symbol("#8#10")),Function,Int64,Val{2}})   # time: 0.001115114
end
