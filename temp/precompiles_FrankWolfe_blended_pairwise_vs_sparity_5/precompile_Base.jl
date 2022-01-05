function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(_similar_shape),UnitRange{Int64},HasShape{1}})   # time: 0.00219437
    Base.precompile(Tuple{Colon,Int64,Int64})   # time: 0.001173309
    Base.precompile(Tuple{typeof(to_shape),Tuple{OneTo{Int64}}})   # time: 0.001102281
end
