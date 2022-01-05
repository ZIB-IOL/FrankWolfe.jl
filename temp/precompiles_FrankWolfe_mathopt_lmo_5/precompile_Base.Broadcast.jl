function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(broadcasted),typeof(-),Any,Vector{Float64}})   # time: 0.013723726
    Base.precompile(Tuple{typeof(getindex),Broadcasted{Nothing, Tuple{OneTo{Int64}}, typeof(*), Tuple{Int64, Broadcasted{DefaultArrayStyle{1}, Nothing, typeof(-), Tuple{Extruded{Vector{Float64}, Tuple{Bool}, Tuple{Int64}}, Extruded{Vector{Float64}, Tuple{Bool}, Tuple{Int64}}}}}},Int64})   # time: 0.001862802
    Base.precompile(Tuple{typeof(broadcasted),typeof(*),Int64,Broadcasted{DefaultArrayStyle{1}, Nothing, typeof(-), Tuple{Vector{Float64}, Vector{Float64}}}})   # time: 0.001518898
end
