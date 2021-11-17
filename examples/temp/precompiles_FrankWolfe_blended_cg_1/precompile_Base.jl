function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(+),Vector{Float64},Vector{Float64}})   # time: 0.023898484
    Base.precompile(Tuple{typeof(/),Vector{Float64},ComplexF64})   # time: 0.01680893
    Base.precompile(Tuple{typeof(/),Vector{Float64},Float64})   # time: 0.011860098
    Base.precompile(Tuple{typeof(+),Any,Float64,Float64,Float64})   # time: 0.007142826
    Base.precompile(Tuple{typeof(getindex),Dict{Int32, Symbol},Int32})   # time: 0.006777885
    Base.precompile(Tuple{typeof(/),ComplexF64,ComplexF64})   # time: 0.005118333
    Base.precompile(Tuple{typeof(sqrt),ComplexF64})   # time: 0.005031558
    Base.precompile(Tuple{Type{NamedTuple{(:t, :primal, :dual, :dual_gap, :time, :x), _A}} where _A<:Tuple{Int64, Any, Any, Float64, Float64, Vector{ComplexF64}},Tuple{Int64, Any, Any, Float64, Float64, Vector{ComplexF64}}})   # time: 0.004657016
    Base.precompile(Tuple{typeof(+),Any,Float64,Any,Any})   # time: 0.003797054
    Base.precompile(Tuple{typeof(copyto_unaliased!),IndexLinear,SubArray{Float64, 1, Matrix{Float64}, Tuple{Slice{OneTo{Int64}}, Int64}, true},IndexLinear,Vector{Float64}})   # time: 0.00364788
    Base.precompile(Tuple{typeof(accumulate_pairwise!),typeof(add_sum),Vector{Float64},Vector{Float64}})   # time: 0.003023585
    Base.precompile(Tuple{typeof(/),Float64,ComplexF64})   # time: 0.002283722
    Base.precompile(Tuple{typeof(_unsafe_getindex),IndexLinear,Matrix{Float64},Int64,Slice{OneTo{Int64}}})   # time: 0.002042922
    Base.precompile(Tuple{typeof(_isdisjoint),Tuple{UInt64, UInt64},Tuple{UInt64, UInt64}})   # time: 0.001794236
    Base.precompile(Tuple{typeof(copyto!),Vector{Int64},UnitRange{Int64}})   # time: 0.001762607
    Base.precompile(Tuple{typeof(vcat),StepRange{Int64, Int64}})   # time: 0.001480878
    Base.precompile(Tuple{typeof(axes),UnitRange{Int64}})   # time: 0.001326169
    Base.precompile(Tuple{typeof(*),Int64,Float64})   # time: 0.001134498
end
