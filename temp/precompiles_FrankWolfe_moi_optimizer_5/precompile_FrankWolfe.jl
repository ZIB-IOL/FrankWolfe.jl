function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(-),ScaledHotVector{Float64},ScaledHotVector})   # time: 0.17576744
    isdefined(FrankWolfe, Symbol("#push_trajectory!#33")) && Base.precompile(Tuple{getfield(FrankWolfe, Symbol("#push_trajectory!#33")),NamedTuple{(:t, :primal, :dual, :dual_gap, :time, :x, :v, :gamma), Tuple{Int64, Float64, Float64, Float64, Float64, ScaledHotVector{Float64}, Vector{Float64}, Float64}}})   # time: 0.04404721
    Base.precompile(Tuple{typeof(perform_line_search),Shortstep{Float64},Int64,Function,Function,SparseVector{Float64, Int64},ScaledHotVector{Float64},Any,Float64,Nothing})   # time: 0.03046424
    Base.precompile(Tuple{typeof(perform_line_search),Shortstep{Float64},Int64,Function,Function,SparseVector{Float64, Int64},Any,Any,Float64,Nothing})   # time: 0.009628172
    Base.precompile(Tuple{Core.kwftype(typeof(print_callback)),NamedTuple{(:print_header,), Tuple{Bool}},typeof(print_callback),Vector{String},String})   # time: 0.004296879
    Base.precompile(Tuple{typeof(perform_line_search),Shortstep{Float64},Int64,Function,Function,SparseVector{Float64, Int64},ScaledHotVector{Float64},Vector{Float64},Float64,Nothing})   # time: 0.004219216
    isdefined(FrankWolfe, Symbol("#push_trajectory!#33")) && Base.precompile(Tuple{getfield(FrankWolfe, Symbol("#push_trajectory!#33")),NamedTuple{(:t, :primal, :dual, :dual_gap, :time, :x, :v, :gamma), _A} where _A<:Tuple{Int64, Any, Any, Any, Float64, ScaledHotVector{Float64}, AbstractVector, Any}})   # time: 0.003979032
    Base.precompile(Tuple{typeof(perform_line_search),Shortstep{Float64},Int64,Function,Function,SparseVector{Float64, Int64},Vector{Float64},Vector{Float64},Float64,Nothing})   # time: 0.003960111
    isdefined(FrankWolfe, Symbol("#push_trajectory!#33")) && Base.precompile(Tuple{getfield(FrankWolfe, Symbol("#push_trajectory!#33")),NamedTuple{(:t, :primal, :dual, :dual_gap, :time, :x, :v, :gamma), _A} where _A<:Tuple{Int64, Float64, Any, Any, Float64, ScaledHotVector{Float64}, AbstractVector, Any}})   # time: 0.003789765
    Base.precompile(Tuple{typeof(fast_dot),Any,SparseVector{Float64, Int64}})   # time: 0.00324892
    Base.precompile(Tuple{typeof(fast_dot),AbstractVector,SparseVector{Float64, Int64}})   # time: 0.002512659
end
