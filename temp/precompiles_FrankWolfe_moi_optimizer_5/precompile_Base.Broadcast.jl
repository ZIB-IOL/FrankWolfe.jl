function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(broadcasted),typeof(-),Any,Vector{Float64}})   # time: 0.037215114
    Base.precompile(Tuple{typeof(getindex),Broadcasted{Nothing, Tuple{OneTo{Int64}}, typeof(*), Tuple{Int64, Broadcasted{DefaultArrayStyle{1}, Nothing, typeof(-), Tuple{Extruded{Vector{Float64}, Tuple{Bool}, Tuple{Int64}}, Extruded{Vector{Float64}, Tuple{Bool}, Tuple{Int64}}}}}},Int64})   # time: 0.005852722
    Base.precompile(Tuple{typeof(broadcasted),typeof(*),Int64,Broadcasted{DefaultArrayStyle{1}, Nothing, typeof(-), Tuple{Vector{Float64}, Vector{Float64}}}})   # time: 0.005789585
    Base.precompile(Tuple{typeof(instantiate),Broadcasted{DefaultArrayStyle{1}, Tuple{OneTo{Int64}}, typeof(*), Tuple{Int64, Broadcasted{DefaultArrayStyle{1}, Nothing, typeof(-), Tuple{Vector{Float64}, Vector{Float64}}}}}})   # time: 0.002776867
end
