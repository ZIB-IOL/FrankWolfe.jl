function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{Type{NamedTuple{(:gamma_max,), T} where T<:Tuple},Tuple{Any}})   # time: 0.006742721
end
