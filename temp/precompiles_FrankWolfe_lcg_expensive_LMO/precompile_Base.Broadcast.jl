function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    isdefined(Base.Broadcast, Symbol("#16#18")) && Base.precompile(Tuple{getfield(Base.Broadcast, Symbol("#16#18")),Float64,Float64,Function,Int64,Val{2}})   # time: 0.001691195
    isdefined(Base.Broadcast, Symbol("#16#18")) && Base.precompile(Tuple{getfield(Base.Broadcast, Symbol("#16#18")),Function,Int64,Val{2}})   # time: 0.001444238
    isdefined(Base.Broadcast, Symbol("#13#14")) && Base.precompile(Tuple{getfield(Base.Broadcast, Symbol("#13#14")),Int64,Float64,Vararg{Any}})   # time: 0.001038531
end
