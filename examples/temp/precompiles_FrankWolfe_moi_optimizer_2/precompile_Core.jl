function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{Type{NamedTuple{(:t, :primal, :dual, :dual_gap, :time, :x, :v, :gamma), T} where T<:Tuple},Core.Tuple{Core.Int64, Core.Any, Core.Any, Core.Float64, Core.Float64, Base.Vector{Core.Float64}, Union{Core.Nothing, Base.Vector{Core.Float64}}, Core.Float64}})   # time: 0.016757805
    Base.precompile(Tuple{Type{NamedTuple{(:t, :primal, :dual, :dual_gap, :time, :x, :v, :gamma), T} where T<:Tuple},Core.Tuple{Core.Int64, Core.Float64, Core.Float64, Core.Float64, Core.Float64, Base.Vector{Core.Float64}, Union{Core.Nothing, Base.Vector{Core.Float64}}, Core.Float64}})   # time: 0.012530591
end
