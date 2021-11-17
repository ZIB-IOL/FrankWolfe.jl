function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{Type{NamedTuple{(:t, :primal, :dual, :dual_gap, :time, :x, :active_set_length, :non_simplex_iter), T} where T<:Tuple},Core.Tuple{Core.Int64, Core.Float64, Core.Any, Core.Any, Core.Float64, Base.Matrix{Core.Float64}, Core.Int64, Core.Int64}})   # time: 0.012427354
end
