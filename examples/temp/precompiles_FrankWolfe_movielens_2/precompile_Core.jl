function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{Type{NamedTuple{(:t, :primal, :dual, :dual_gap, :time, :cache_size, :x, :v, :gamma), T} where T<:Tuple},Tuple{Int64, Any, Any, Any, Float64, Any, Any, Any, Any}})   # time: 0.00603553
    Base.precompile(Tuple{Type{NamedTuple{(:t, :primal, :dual, :dual_gap, :time, :cache_size, :x, :v, :gamma), T} where T<:Tuple},Tuple{Int64, Any, Any, Any, Float64, Int64, Any, Any, Any}})   # time: 0.005259456
    Base.precompile(Tuple{Type{NamedTuple{(:threshold, :greedy), T} where T<:Tuple},Tuple{Any, Bool}})   # time: 0.004146817
end
