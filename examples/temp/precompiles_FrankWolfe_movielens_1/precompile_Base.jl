function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(iterate),StepRange{Int64, Int64},Int64})   # time: 0.04152251
    Base.precompile(Tuple{typeof(-),Matrix{BigFloat},Matrix{BigFloat}})   # time: 0.031219274
    Base.precompile(Tuple{typeof(*),Float64,Matrix{BigFloat}})   # time: 0.024897946
    Base.precompile(Tuple{typeof(-),Matrix{BigFloat},Matrix{Float64}})   # time: 0.023480598
    Base.precompile(Tuple{typeof(*),BigFloat,Matrix{Float64}})   # time: 0.0224517
    Base.precompile(Tuple{typeof(-),Matrix{Float64},Matrix{BigFloat}})   # time: 0.022103604
    Base.precompile(Tuple{typeof(*),BigFloat,Matrix{BigFloat}})   # time: 0.019227821
    Base.precompile(Tuple{typeof(getindex),Dict{Int32, Symbol},Int32})   # time: 0.006548227
    Base.precompile(Tuple{typeof(string),Int64})   # time: 0.005263785
    Base.precompile(Tuple{typeof(vect),String,String,String,String,String,String,String})   # time: 0.004195991
    Base.precompile(Tuple{typeof(structdiff),NamedTuple{(:gamma_max,), _A} where _A<:Tuple{Any},Type{NamedTuple{(:eta, :tau, :gamma_max, :upgrade_accuracy), T} where T<:Tuple}})   # time: 0.003949376
    Base.precompile(Tuple{typeof(push!),Vector{Tuple{Int64, Float64, Float64, Float64, Float64, Float64}},Tuple})   # time: 0.002229187
    Base.precompile(Tuple{typeof(getindex),Tuple,UnitRange{Int64}})   # time: 0.00188948
    Base.precompile(Tuple{typeof(_array_for),Type{Any},UnitRange{Int64},HasShape{1}})   # time: 0.00173318
    Base.precompile(Tuple{typeof(max),BigFloat,Float64})   # time: 0.001640779
    Base.precompile(Tuple{typeof(copyto!),Vector{Int64},Int64,UnitRange{Int64}})   # time: 0.001207694
end
