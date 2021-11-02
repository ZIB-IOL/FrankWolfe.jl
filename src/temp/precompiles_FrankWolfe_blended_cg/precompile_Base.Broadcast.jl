function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    isdefined(Base.Broadcast, Symbol("#5#6")) && Base.precompile(Tuple{getfield(Base.Broadcast, Symbol("#5#6")),Float64,Int64,Vararg{Any, N} where N})   # time: 0.0043606
    isdefined(Base.Broadcast, Symbol("#8#10")) && Base.precompile(Tuple{getfield(Base.Broadcast, Symbol("#8#10")),Int64,Float64,Float64,Float64})   # time: 0.001560809
    Base.precompile(Tuple{typeof(getindex),Broadcasted{Nothing, Tuple{OneTo{Int64}}, typeof(identity), Tuple{Extruded{Vector{Float64}, Tuple{Bool}, Tuple{Int64}}}},Int64})   # time: 0.001442778
end
