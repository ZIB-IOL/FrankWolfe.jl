function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    isdefined(FrankWolfe, Symbol("#push_trajectory!#33")) && Base.precompile(Tuple{getfield(FrankWolfe, Symbol("#push_trajectory!#33")),NamedTuple{(:t, :primal, :dual, :dual_gap, :time, :x, :v, :gamma), Tuple{Int64, Float64, Float64, Float64, Float64, Vector{Float64}, Vector{Float64}, Float64}}})   # time: 0.14186335
    Base.precompile(Tuple{Core.kwftype(typeof(print_callback)),NamedTuple{(:print_header,), Tuple{Bool}},typeof(print_callback),NTuple{8, String},String})   # time: 0.104074836
    Base.precompile(Tuple{Core.kwftype(typeof(print_callback)),NamedTuple{(:print_footer,), Tuple{Bool}},typeof(print_callback),Nothing,String})   # time: 0.007996668
    Base.precompile(Tuple{typeof(perform_line_search),Nonconvex{Float64},Int64,Nothing,Nothing,Vector{Float64},Vector{Float64},Vector{Float64},Float64,Nothing})   # time: 0.002293631
end
