function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(broadcasted),typeof(*),Int64,Broadcasted{DefaultArrayStyle{2}, Nothing, typeof(-), Tuple{Matrix{Float64}, Matrix{Float64}}}})   # time: 0.003563771
    Base.precompile(Tuple{typeof(broadcasted),typeof(/),Broadcasted{DefaultArrayStyle{2}, Nothing, typeof(*), Tuple{Int64, Broadcasted{DefaultArrayStyle{2}, Nothing, typeof(-), Tuple{Matrix{Float64}, Matrix{Float64}}}}},Any})   # time: 0.002325
    isdefined(Base.Broadcast, Symbol("#13#14")) && Base.precompile(Tuple{getfield(Base.Broadcast, Symbol("#13#14")),Float64,Int64,Vararg{Any}})   # time: 0.001596071
    isdefined(Base.Broadcast, Symbol("#16#18")) && Base.precompile(Tuple{getfield(Base.Broadcast, Symbol("#16#18")),Int64,Float64,Float64,Float64})   # time: 0.001534476
end
