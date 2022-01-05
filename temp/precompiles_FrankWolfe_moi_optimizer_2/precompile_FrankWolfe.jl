function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    isdefined(FrankWolfe, Symbol("#push_trajectory!#33")) && Base.precompile(Tuple{getfield(FrankWolfe, Symbol("#push_trajectory!#33")),NamedTuple{(:t, :primal, :dual, :dual_gap, :time, :x, :v, :gamma), Tuple{Int64, Float64, Float64, Float64, Float64, Vector{Float64}, Vector{Float64}, Float64}}})   # time: 0.119292945
    isdefined(FrankWolfe, Symbol("#push_trajectory!#33")) && Base.precompile(Tuple{getfield(FrankWolfe, Symbol("#push_trajectory!#33")),NamedTuple{(:t, :primal, :dual, :dual_gap, :time, :x, :v, :gamma), _A} where _A<:Tuple{Int64, Any, Any, Float64, Float64, Vector{Float64}, Vector{Float64}, Float64}})   # time: 0.003231667
    Base.precompile(Tuple{typeof(perform_line_search),Shortstep{Float64},Int64,Function,Function,Vector{Float64},Vector{Float64},Vector{Float64},Float64,Nothing})   # time: 0.003157497
    isdefined(FrankWolfe, Symbol("#81#82")) && Base.precompile(Tuple{getfield(FrankWolfe, Symbol("#81#82")),Tuple{Float64, MathOptInterface.VariableIndex}})   # time: 0.002547302
    Base.precompile(Tuple{Core.kwftype(typeof(print_callback)),NamedTuple{(:print_header,), Tuple{Bool}},typeof(print_callback),Vector{String},String})   # time: 0.002528235
end
