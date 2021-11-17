function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{Core.kwftype(typeof(frank_wolfe)),NamedTuple{(:max_iteration, :line_search, :L, :print_iter, :emphasis, :verbose, :trajectory), Tuple{Int64, Shortstep, Int64, Float64, Emphasis, Bool, Bool}},typeof(frank_wolfe),Function,Function,ProbabilitySimplexOracle{Float64},ScaledHotVector{Float64}})   # time: 0.6060258
    Base.precompile(Tuple{typeof(-),ScaledHotVector{Float64},Vector{Float64}})   # time: 0.03285388
    isdefined(FrankWolfe, Symbol("#push_trajectory!#9")) && Base.precompile(Tuple{getfield(FrankWolfe, Symbol("#push_trajectory!#9")),NamedTuple{(:t, :primal, :dual, :dual_gap, :time, :x, :v, :gamma), Tuple{Int64, Float64, Float64, Float64, Float64, SparseVector{Float64, Int64}, ScaledHotVector{Float64}, Float64}}})   # time: 0.01333194
    isdefined(FrankWolfe, Symbol("#push_trajectory!#9")) && Base.precompile(Tuple{getfield(FrankWolfe, Symbol("#push_trajectory!#9")),NamedTuple{(:t, :primal, :dual, :dual_gap, :time, :x, :v, :gamma), Tuple{Int64, Float64, Float64, Float64, Float64, ScaledHotVector{Float64}, ScaledHotVector{Float64}, Float64}}})   # time: 0.012246269
    isdefined(FrankWolfe, Symbol("#push_trajectory!#9")) && Base.precompile(Tuple{getfield(FrankWolfe, Symbol("#push_trajectory!#9")),NamedTuple{(:t, :primal, :dual, :dual_gap, :time, :x, :v, :gamma), _A} where _A<:Tuple{Int64, Float64, Float64, Float64, Float64, Union{ScaledHotVector{Float64}, SparseVector{Float64, Int64}}, ScaledHotVector{Float64}, Float64}})   # time: 0.005551316
    Base.precompile(Tuple{Core.kwftype(typeof(print_callback)),NamedTuple{(:print_header,), Tuple{Bool}},typeof(print_callback),Vector{String},String})   # time: 0.001894134
end
