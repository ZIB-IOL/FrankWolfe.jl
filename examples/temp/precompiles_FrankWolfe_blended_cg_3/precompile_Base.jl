function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(getindex),Dict{Int32, Symbol},Int32})   # time: 0.005594544
    Base.precompile(Tuple{typeof(axes),UnitRange{Int64}})   # time: 0.001204052
end
