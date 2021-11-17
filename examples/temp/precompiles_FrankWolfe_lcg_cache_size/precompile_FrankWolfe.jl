function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(print_callback),Tuple{String, String, Float64, Any, Any, Float64, Float64, Any},String})   # time: 0.05358488
    Base.precompile(Tuple{Type{MultiCacheLMO{_A, KSparseLMO{Float64}, _B}} where {_A, _B},KSparseLMO{Float64}})   # time: 0.007007704
    Base.precompile(Tuple{Type{VectorCacheLMO{KSparseLMO{Float64}, _A}} where _A,KSparseLMO{Float64}})   # time: 0.006266348
    Base.precompile(Tuple{typeof(print_callback),Tuple{String, String, Float64, Float64, Float64, Float64, Float64, Int64},String})   # time: 0.005020738
    isdefined(FrankWolfe, Symbol("#push_trajectory!#9")) && Base.precompile(Tuple{getfield(FrankWolfe, Symbol("#push_trajectory!#9")),NamedTuple{(:t, :primal, :dual, :dual_gap, :time, :cache_size, :x, :v, :gamma), _A} where _A<:Tuple{Int64, Float64, Any, Any, Float64, Any, SparseVector{Float64, Int64}, Any, Any}})   # time: 0.00240747
    Base.precompile(Tuple{typeof(fast_dot),SparseVector{BigFloat, Int64},Any})   # time: 0.002146262
    Base.precompile(Tuple{typeof(fast_dot),Any,SparseVector{Float64, Int64}})   # time: 0.002042688
    Base.precompile(Tuple{typeof(print_callback),Tuple{String, String, Any, Any, Any, Float64, Float64, Any},String})   # time: 0.001693418
    Base.precompile(Tuple{Core.kwftype(typeof(print_callback)),NamedTuple{(:print_header,), Tuple{Bool}},typeof(print_callback),Vector{String},String})   # time: 0.001017992
end
