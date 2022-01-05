function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{Type{NamedTuple{(:t, :primal, :dual, :dual_gap, :time, :x, :v, :gamma)}},Tuple{Core.Int64, Core.Any, Core.Any, Core.Float64, Core.Float64, Base.Vector{Core.Float64}, Base.Vector{Core.Float64}, Core.Float64}})   # time: 0.002782588
end
