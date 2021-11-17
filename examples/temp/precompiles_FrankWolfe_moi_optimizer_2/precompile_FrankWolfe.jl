function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    isdefined(FrankWolfe, Symbol("#push_trajectory!#9")) && Base.precompile(Tuple{getfield(FrankWolfe, Symbol("#push_trajectory!#9")),NamedTuple{(:t, :primal, :dual, :dual_gap, :time, :x, :v, :gamma), Tuple{Int64, Float64, Float64, Float64, Float64, Vector{Float64}, Vector{Float64}, Float64}}})   # time: 0.020447256
    isdefined(FrankWolfe, Symbol("#push_trajectory!#9")) && Base.precompile(Tuple{getfield(FrankWolfe, Symbol("#push_trajectory!#9")),NamedTuple{(:t, :primal, :dual, :dual_gap, :time, :x, :v, :gamma), _A} where _A<:Tuple{Int64, Float64, Float64, Float64, Float64, Vector{Float64}, Union{Nothing, Vector{Float64}}, Float64}})   # time: 0.005750711
    isdefined(FrankWolfe, Symbol("#push_trajectory!#9")) && Base.precompile(Tuple{getfield(FrankWolfe, Symbol("#push_trajectory!#9")),NamedTuple{(:t, :primal, :dual, :dual_gap, :time, :x, :v, :gamma), _A} where _A<:Tuple{Int64, Any, Any, Float64, Float64, Vector{Float64}, Union{Nothing, Vector{Float64}}, Float64}})   # time: 0.005327483
    Base.precompile(Tuple{typeof(line_search_wrapper),Shortstep,Int64,Function,Function,Vector{Float64},Vector{Float64},Vector{Float64},Float64,Int64,Int64,Float64,Int64,Float64})   # time: 0.003564647
    Base.precompile(Tuple{Core.kwftype(typeof(print_callback)),NamedTuple{(:print_header,), Tuple{Bool}},typeof(print_callback),Vector{String},String})   # time: 0.002068177
    isdefined(FrankWolfe, Symbol("#54#55")) && Base.precompile(Tuple{getfield(FrankWolfe, Symbol("#54#55")),Tuple{Float64, MathOptInterface.VariableIndex}})   # time: 0.001967329
end
