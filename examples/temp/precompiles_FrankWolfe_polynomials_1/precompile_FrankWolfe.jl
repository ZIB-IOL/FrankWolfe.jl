function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(line_search_wrapper),Adaptive,Int64,Function,Function,SparseVector{Float64, Int64},SparseVector{Float64, Int64},Vector{Float64},Float64,Float64,Int64,Float64,Int64,Float64})   # time: 0.38120192
    Base.precompile(Tuple{Core.kwftype(typeof(print_callback)),NamedTuple{(:print_header,), Tuple{Bool}},typeof(print_callback),NTuple{8, String},String})   # time: 0.1179173
    Base.precompile(Tuple{typeof(line_search_wrapper),Adaptive,Int64,Function,Function,SparseVector{Float64, Int64},SparseVector{Float64, Int64},Vector{Float64},Float64,Float64,Int64,Float64,Int64,Int64})   # time: 0.016679563
    Base.precompile(Tuple{typeof(print_callback),Tuple{String, String, Float64, Float64, Float64, Float64, Float64, Int64},String})   # time: 0.003570676
end
