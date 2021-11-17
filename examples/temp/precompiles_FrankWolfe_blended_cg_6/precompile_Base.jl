function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(getindex),Dict{Int32, Symbol},Int32})   # time: 0.005632849
    Base.precompile(Tuple{typeof(axes),UnitRange{Int64}})   # time: 0.001131769
end
