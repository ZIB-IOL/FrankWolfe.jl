function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{Type{NamedTuple{(:t, :primal, :dual, :dual_gap, :time, :x, :active_set_length, :non_simplex_iter)}},Tuple{Core.Int64, Core.Any, Core.Any, Core.Float64, Core.Float64, Base.Vector{Core.Float64}, Core.Int64, Core.Int64}})   # time: 0.006745652
end
