function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(active_set_update!),ActiveSet{ScaledHotVector{Float64}, Float64, Vector{Float64}},Float64,ScaledHotVector{Float64},Bool,Nothing})   # time: 0.16729474
    Base.precompile(Tuple{Type{ActiveSet},Vector{Tuple{Float64, ScaledHotVector{Float64}}}})   # time: 0.14734647
    Base.precompile(Tuple{Core.kwftype(typeof(print_callback)),NamedTuple{(:print_header,), Tuple{Bool}},typeof(print_callback),NTuple{8, String},String})   # time: 0.0798516
    Base.precompile(Tuple{typeof(perform_line_search),Adaptive{Float64, Int64},Int64,Function,Function,Vector{Float64},Vector{Float64},Vector{Float64},Float64,Vector{Float64}})   # time: 0.05143009
    Base.precompile(Tuple{Core.kwftype(typeof(lazy_afw_step)),NamedTuple{(:lazy_tolerance,), Tuple{Float64}},typeof(lazy_afw_step),Vector{Float64},Vector{Float64},LpNormLMO{Float64, 1},ActiveSet{ScaledHotVector{Float64}, Float64, Vector{Float64}},Float64})   # time: 0.0172512
    Base.precompile(Tuple{Core.kwftype(typeof(Type)),NamedTuple{(:L_est,), Tuple{Float64}},Type{Adaptive}})   # time: 0.010198866
    Base.precompile(Tuple{Core.kwftype(typeof(print_callback)),NamedTuple{(:print_footer,), Tuple{Bool}},typeof(print_callback),Nothing,String})   # time: 0.006230425
    Base.precompile(Tuple{typeof(active_set_update!),ActiveSet{ScaledHotVector{Float64}, Float64, Vector{Float64}},Float64,ScaledHotVector{Float64},Bool,Int64})   # time: 0.005069646
    Base.precompile(Tuple{typeof(print_callback),Tuple{String, String, Any, Any, Float64, Float64, Float64, Int64},String})   # time: 0.004002943
    Base.precompile(Tuple{typeof(afw_step),Vector{Float64},Vector{Float64},LpNormLMO{Float64, 1},ActiveSet{ScaledHotVector{Float64}, Float64, Vector{Float64}}})   # time: 0.00219724
end
