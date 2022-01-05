function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(print_callback),Tuple{String, String, Float64, Any, Any, Float64, Float64},String})   # time: 0.00497477
    Base.precompile(Tuple{typeof(print_callback),Tuple{String, String, Any, Any, Any, Float64, Float64},String})   # time: 0.003270714
    Base.precompile(Tuple{typeof(fast_dot),AbstractVector,Vector{Float64}})   # time: 0.002228764
    Base.precompile(Tuple{typeof(perform_line_search),Shortstep{Float64},Int64,Function,Function,Vector{Float64},Vector{Float64},Vector{Float64},Float64,Nothing})   # time: 0.002015472
    isdefined(FrankWolfe, Symbol("#push_trajectory!#33")) && Base.precompile(Tuple{getfield(FrankWolfe, Symbol("#push_trajectory!#33")),NamedTuple{(:t, :primal, :dual, :dual_gap, :time, :x, :v, :gamma), _A} where _A<:Tuple{Int64, Any, Any, Any, Float64, Vector{Float64}, AbstractVector, Float64}})   # time: 0.001918669
    isdefined(FrankWolfe, Symbol("#push_trajectory!#33")) && Base.precompile(Tuple{getfield(FrankWolfe, Symbol("#push_trajectory!#33")),NamedTuple{(:t, :primal, :dual, :dual_gap, :time, :x, :v, :gamma), _A} where _A<:Tuple{Int64, Float64, Any, Any, Float64, Vector{Float64}, AbstractVector, Float64}})   # time: 0.001615074
    Base.precompile(Tuple{Core.kwftype(typeof(print_callback)),NamedTuple{(:print_header,), Tuple{Bool}},typeof(print_callback),Vector{String},String})   # time: 0.001332354
end
