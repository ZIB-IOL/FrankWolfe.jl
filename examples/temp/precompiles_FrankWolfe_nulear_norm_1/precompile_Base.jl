function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(-),Matrix{Float64},Matrix{BigFloat}})   # time: 0.04074972
    Base.precompile(Tuple{typeof(-),Matrix{BigFloat},Matrix{BigFloat}})   # time: 0.039427374
    Base.precompile(Tuple{typeof(*),BigFloat,Matrix{Float64}})   # time: 0.033391014
    Base.precompile(Tuple{typeof(-),Matrix{BigFloat},Matrix{Float64}})   # time: 0.032640014
    Base.precompile(Tuple{typeof(*),Float64,Matrix{BigFloat}})   # time: 0.03222888
    Base.precompile(Tuple{typeof(*),BigFloat,Matrix{BigFloat}})   # time: 0.026948957
    Base.precompile(Tuple{typeof(-),Matrix{Float64},Matrix{Float64}})   # time: 0.013767458
    Base.precompile(Tuple{typeof(getindex),Dict{Int32, Symbol},Int32})   # time: 0.010856946
    Base.precompile(Tuple{typeof(structdiff),NamedTuple{(:gamma_max,), _A} where _A<:Tuple{Any},Type{NamedTuple{(:eta, :tau, :gamma_max, :upgrade_accuracy), T} where T<:Tuple}})   # time: 0.008362804
    Base.precompile(Tuple{typeof(string),Int64})   # time: 0.008022962
    Base.precompile(Tuple{typeof(sum),Generator{Vector{Tuple{Int64, Int64}}, _A} where _A})   # time: 0.004390082
    Base.precompile(Tuple{typeof(copyto!),Vector{Int64},Int64,UnitRange{Int64}})   # time: 0.004255007
    Base.precompile(Tuple{typeof(max),BigFloat,Float64})   # time: 0.002325538
    Base.precompile(Tuple{typeof(mod),Int64,Float64})   # time: 0.00164865
    Base.precompile(Tuple{typeof(>),BigFloat,Float64})   # time: 0.001335763
    Base.precompile(Tuple{typeof(zeros),Type{Int64},Int64})   # time: 0.001285967
    Base.precompile(Tuple{BottomRF{typeof(add_sum)},BigFloat,BigFloat})   # time: 0.001126312
    Base.precompile(Tuple{typeof(/),UInt64,Float64})   # time: 0.001007027
    Base.precompile(Tuple{typeof(min),BigFloat,Float64})   # time: 0.001004932
end
