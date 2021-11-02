function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(broadcasted),Function,Matrix{Float64},Broadcasted{DefaultArrayStyle{2}, Nothing, typeof(*), Tuple{Float64, Matrix{Float64}}}})   # time: 0.006898905
    Base.precompile(Tuple{typeof(broadcasted),Function,Float64,Matrix{Float64}})   # time: 0.006467426
    Base.precompile(Tuple{typeof(broadcasted),typeof(+),Int64,UnitRange{Int64}})   # time: 0.002460431
    Base.precompile(Tuple{Type{Broadcasted{DefaultArrayStyle{2}, Axes, F, Args} where {Axes, F, Args<:Tuple}},typeof(-),Tuple{Matrix{Float64}, Broadcasted{DefaultArrayStyle{2}, Nothing, typeof(*), Tuple{Float64, Matrix{Float64}}}}})   # time: 0.002269905
    Base.precompile(Tuple{typeof(_broadcast_getindex),Extruded{Matrix{Float64}, Tuple{Bool, Bool}, Tuple{Int64, Int64}},CartesianIndex{2}})   # time: 0.001516878
    Base.precompile(Tuple{typeof(broadcasted),typeof(identity),Int64})   # time: 0.001481957
end
