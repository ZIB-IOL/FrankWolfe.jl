function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{Core.kwftype(typeof(frank_wolfe)),NamedTuple{(:max_iteration, :line_search, :L, :print_iter, :emphasis, :verbose, :trajectory), Tuple{Int64, Shortstep, Int64, Float64, Emphasis, Bool, Bool}},typeof(frank_wolfe),Function,Function,MathOptLMO{MathOptInterface.Utilities.CachingOptimizer{MathOptInterface.AbstractOptimizer, MathOptInterface.Utilities.UniversalFallback{MathOptInterface.Utilities.GenericModel{Float64, MathOptInterface.Utilities.ModelFunctionConstraints{Float64}}}}},Vector{Float64}})   # time: 5.6999826
    Base.precompile(Tuple{typeof(print_callback),Tuple{String, String, Float64, Any, Any, Float64, Float64},String})   # time: 0.069496654
    Base.precompile(Tuple{Core.kwftype(typeof(print_callback)),NamedTuple{(:print_header,), Tuple{Bool}},typeof(print_callback),Vector{String},String})   # time: 0.001010237
end
