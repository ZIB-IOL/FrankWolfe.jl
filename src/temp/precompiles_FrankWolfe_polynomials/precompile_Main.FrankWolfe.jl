function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{Core.kwftype(typeof(print_callback)),NamedTuple{(:print_header,), Tuple{Bool}},typeof(print_callback),NTuple{8, String},String})   # time: 0.10320469
    Base.precompile(Tuple{typeof(active_set_update!),ActiveSet{ScaledHotVector{Float64}, Float64, SparseVector{Float64, Int64}},Float64,ScaledHotVector{Float64},Bool,Int64})   # time: 0.02599051
    Base.precompile(Tuple{typeof(active_set_update!),ActiveSet{ScaledHotVector{Float64}, Float64, SparseVector{Float64, Int64}},BigFloat,ScaledHotVector{Float64},Bool,Int64})   # time: 0.025380524
    Base.precompile(Tuple{typeof(active_set_update!),ActiveSet{ScaledHotVector{Float64}, Float64, SparseVector{Float64, Int64}},Int64,ScaledHotVector{Float64},Bool,Int64})   # time: 0.023262447
    Base.precompile(Tuple{typeof(line_search_wrapper),Adaptive,Int64,Function,Function,SparseVector{Float64, Int64},SparseVector{Float64, Int64},Vector{Float64},Float64,Float64,Int64,Float64,Int64,Float64})   # time: 0.017802905
    Base.precompile(Tuple{typeof(line_search_wrapper),Adaptive,Int64,Function,Function,SparseVector{Float64, Int64},Vector{Float64},Vector{Float64},Float64,Float64,Int64,Float64,Int64,Float64})   # time: 0.014568289
    Base.precompile(Tuple{typeof(line_search_wrapper),Adaptive,Int64,Function,Function,SparseVector{Float64, Int64},Vector{Float64},Vector{Float64},Float64,Float64,Int64,Float64,Int64,Int64})   # time: 0.01240859
    Base.precompile(Tuple{Core.kwftype(typeof(lazy_afw_step)),NamedTuple{(:K,), Tuple{Float64}},typeof(lazy_afw_step),SparseVector{Float64, Int64},Vector{Float64},LpNormLMO{Float64, 1},ActiveSet{ScaledHotVector{Float64}, Float64, SparseVector{Float64, Int64}},Float64})   # time: 0.01069944
    Base.precompile(Tuple{typeof(line_search_wrapper),Adaptive,Int64,Function,Function,SparseVector{Float64, Int64},SparseVector{Float64, Int64},Vector{Float64},Float64,Float64,Int64,Float64,Int64,Int64})   # time: 0.010481326
    Base.precompile(Tuple{typeof(print_callback),Tuple{String, String, Float64, Float64, Float64, Float64, Float64, Int64},String})   # time: 0.004592817
    Base.precompile(Tuple{typeof(afw_step),SparseVector{Float64, Int64},Vector{Float64},LpNormLMO{Float64, 1},ActiveSet{ScaledHotVector{Float64}, Float64, SparseVector{Float64, Int64}}})   # time: 0.002910835
    Base.precompile(Tuple{typeof(fw_step),SparseVector{Float64, Int64},Vector{Float64},LpNormLMO{Float64, 1}})   # time: 0.001129889
end
