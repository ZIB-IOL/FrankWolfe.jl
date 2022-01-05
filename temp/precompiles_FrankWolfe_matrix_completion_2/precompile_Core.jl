function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{Type{NamedTuple{(:t, :primal, :dual, :dual_gap, :time, :cache_size, :x, :v, :gamma)}},Tuple{Core.Int64, Core.Any, Core.Any, Core.Any, Core.Float64, Core.Int64, Base.Matrix{Core.Float64}, Core.Any, Core.Float64}})   # time: 0.007914646
    Base.precompile(Tuple{Type{NamedTuple{(:threshold, :greedy)}},Tuple{Any, Bool}})   # time: 0.006180495
    Base.precompile(Tuple{Type{NamedTuple{(:t, :primal, :dual, :dual_gap, :time, :cache_size, :x, :v, :gamma)}},Tuple{Core.Int64, Core.Float64, Core.Any, Core.Any, Core.Float64, Core.Any, Base.Matrix{Core.Float64}, Core.Any, Core.Float64}})   # time: 0.005891779
end
