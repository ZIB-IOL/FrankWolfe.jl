function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{Core.kwftype(typeof(lp_separation_oracle)),NamedTuple{(:inplace_loop, :force_fw_step), Tuple{Bool, Bool}},typeof(lp_separation_oracle),LpNormLMO{Float64, 1},ActiveSet{ScaledHotVector{Float64}, Float64, Vector{Float64}},SparseVector{Float64, Int64},Float64,Float64})   # time: 0.13437502
    Base.precompile(Tuple{Core.kwftype(typeof(perform_line_search)),NamedTuple{(:should_upgrade,), Tuple{Val{true}}},typeof(perform_line_search),Adaptive{Float64, Int64},Int64,Function,Function,SparseVector{Float64, Int64},Vector{Float64},Vector{Float64},Float64,Vector{Float64}})   # time: 0.10003789
    Base.precompile(Tuple{typeof(active_set_initialize!),ActiveSet{ScaledHotVector{Float64}, Float64, Vector{Float64}},ScaledHotVector{Float64}})   # time: 0.02555044
    Base.precompile(Tuple{typeof(perform_line_search),Adaptive{Float64, Int64},Int64,Function,Function,SparseVector{Float64, Int64},Vector{Float64},Vector{Float64},Float64,Vector{Float64}})   # time: 0.009839748
    Base.precompile(Tuple{Core.kwftype(typeof(print_callback)),NamedTuple{(:print_header,), Tuple{Bool}},typeof(print_callback),NTuple{9, String},String})   # time: 0.005385012
    isdefined(FrankWolfe, Symbol("#113#115")) && Base.precompile(Tuple{getfield(FrankWolfe, Symbol("#113#115")),ScaledHotVector{Float64}})   # time: 0.005029945
    Base.precompile(Tuple{typeof(fast_dot),Vector{BigFloat},Vector{BigFloat}})   # time: 0.003462153
    Base.precompile(Tuple{typeof(compute_extreme_point),LpNormLMO{Float64, 1},SparseVector{Float64, Int64}})   # time: 0.00309479
    Base.precompile(Tuple{Core.kwftype(typeof(active_set_cleanup!)),NamedTuple{(:weight_purge_threshold,), Tuple{Float64}},typeof(active_set_cleanup!),ActiveSet{ScaledHotVector{Float64}, Float64, Vector{Float64}}})   # time: 0.001547255
    Base.precompile(Tuple{typeof(print_callback),Tuple{String, String, Any, Any, Float64, Float64, Float64, Int64, Int64},String})   # time: 0.001473014
end
