function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(getindex),Dict{Int32, Symbol},Int32})   # time: 0.005284723
    Base.precompile(Tuple{typeof(copyto_unaliased!),IndexLinear,SubArray{Int64, 1, Vector{Int64}, Tuple{UnitRange{Int64}}, true},IndexLinear,Vector{Int64}})   # time: 0.002952741
end
