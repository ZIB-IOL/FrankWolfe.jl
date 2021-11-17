function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(Base.to_index),DenseAxisArray,Tuple{Any}})   # time: 0.012587019
    Base.precompile(Tuple{Type{DenseAxisArray},Core.Array{T, N},Any,Tuple{Vararg{_AxisLookup, N}} where N})   # time: 0.003936612
    Base.precompile(Tuple{typeof(_broadcast_axes),Tuple{Base.RefValue, Any, Any}})   # time: 0.001236728
    Base.precompile(Tuple{typeof(_broadcast_axes),Tuple{Any, Any}})   # time: 0.001083572
end
