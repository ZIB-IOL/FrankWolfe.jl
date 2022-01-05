function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(_similar_shape),UnitRange{Int64},HasShape{1}})   # time: 0.012077819
    Base.precompile(Tuple{Colon,Int64,Int64})   # time: 0.001404905
end
