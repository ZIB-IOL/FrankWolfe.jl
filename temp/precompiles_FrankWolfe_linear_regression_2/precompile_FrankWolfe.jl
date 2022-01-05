function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{Core.kwftype(typeof(print_callback)),NamedTuple{(:print_header,), Tuple{Bool}},typeof(print_callback),NTuple{8, String},String})   # time: 0.014574182
    isdefined(FrankWolfe, Symbol("#push_trajectory!#33")) && Base.precompile(Tuple{getfield(FrankWolfe, Symbol("#push_trajectory!#33")),NamedTuple{(:t, :primal, :dual, :dual_gap, :time, :x, :v, :gamma), Tuple{Int64, Float64, Float64, Float64, Float64, Vector{Float64}, Vector{Float64}, Float64}}})   # time: 0.00597737
    Base.precompile(Tuple{typeof(fast_dot),Vector{Float64},Int64})   # time: 0.004132393
    Base.precompile(Tuple{typeof(perform_line_search),Nonconvex{Float64},Int64,Nothing,Nothing,Vector{Float64},Vector{Float64},Vector{Float64},Float64,Nothing})   # time: 0.002333928
    Base.precompile(Tuple{typeof(compute_extreme_point),LpNormLMO{Float64, 2},Int64})   # time: 0.001986683
end
