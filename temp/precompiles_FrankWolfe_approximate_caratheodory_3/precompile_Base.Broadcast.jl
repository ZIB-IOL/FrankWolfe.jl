function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(instantiate),Broadcasted{Style{Tuple}, Tuple{OneTo{Int64}}}})   # time: 0.002993598
    Base.precompile(Tuple{typeof(instantiate),Broadcasted{_A, Tuple{OneTo{Int64}}} where _A<:Union{Nothing, BroadcastStyle}})   # time: 0.002148378
    Base.precompile(Tuple{Type{Broadcasted{_A}} where _A,Any,Tuple,Tuple{OneTo{Int64}}})   # time: 0.001131983
end
