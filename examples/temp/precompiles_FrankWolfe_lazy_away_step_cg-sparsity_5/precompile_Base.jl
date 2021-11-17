function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(axes),UnitRange{Int64}})   # time: 0.002453715
end
