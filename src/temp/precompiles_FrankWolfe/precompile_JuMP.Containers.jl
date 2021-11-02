function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{Type{DenseAxisArray},Core.Array{T, N},Any,Tuple{Vararg{_AxisLookup, N}} where N})   # time: 0.003218814
end
