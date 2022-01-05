function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{Type{NamedTuple{(:t, :primal, :dual, :dual_gap, :time, :x, :v, :gamma)}},Tuple{Core.Int64, Core.Any, Core.Any, Core.Any, Core.Float64, Core.Any, Base.AbstractVector, Core.Any}})   # time: 0.005915032
end
