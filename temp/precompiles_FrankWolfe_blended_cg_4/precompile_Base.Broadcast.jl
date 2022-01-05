function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    isdefined(Base.Broadcast, Symbol("#16#18")) && Base.precompile(Tuple{getfield(Base.Broadcast, Symbol("#16#18")),Int64,Float64,Float64,Float64})   # time: 0.001087171
    isdefined(Base.Broadcast, Symbol("#13#14")) && Base.precompile(Tuple{getfield(Base.Broadcast, Symbol("#13#14")),Float64,Int64,Vararg{Any}})   # time: 0.00106252
end
