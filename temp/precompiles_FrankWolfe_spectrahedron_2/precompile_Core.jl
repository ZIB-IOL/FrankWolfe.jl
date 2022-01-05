function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{Type{NamedTuple{(:t, :primal, :dual, :dual_gap, :time, :cache_size, :x, :v, :gamma)}},Tuple{Core.Int64, Core.Float64, Core.Any, Core.Any, Core.Float64, Core.Any, Base.Matrix{Core.Float64}, Core.Any, Core.Float64}})   # time: 0.001261885
end
