function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(instantiate),Broadcasted{Style{Tuple}, Tuple{OneTo{Int64}}, _B, _C} where {_B, _C<:Tuple}})   # time: 0.003177451
    Base.precompile(Tuple{typeof(instantiate),Broadcasted{_A, Tuple{OneTo{Int64}}, _B, _C} where {_A<:Union{Nothing, BroadcastStyle}, _B, _C<:Tuple}})   # time: 0.00258465
    Base.precompile(Tuple{Type{Broadcasted{_A, Axes, F, Args} where {Axes, F, Args<:Tuple}} where _A,Any,Tuple,Tuple{OneTo{Int64}}})   # time: 0.001464899
end
