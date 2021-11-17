function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{Type{NamedTuple{(:gamma_max, :upgrade_accuracy), T} where T<:Tuple},Tuple{Float64, Bool}})   # time: 0.005453924
end
