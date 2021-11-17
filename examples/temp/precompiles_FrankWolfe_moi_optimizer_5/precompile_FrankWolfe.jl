function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{Core.kwftype(typeof(frank_wolfe)),NamedTuple{(:max_iteration, :line_search, :L, :print_iter, :emphasis, :verbose, :trajectory), Tuple{Int64, Shortstep, Int64, Float64, Emphasis, Bool, Bool}},typeof(frank_wolfe),Function,Function,MathOptLMO{MathOptInterface.Utilities.CachingOptimizer{MathOptInterface.AbstractOptimizer, MathOptInterface.Utilities.UniversalFallback{MathOptInterface.Utilities.GenericModel{Float64, MathOptInterface.Utilities.ModelFunctionConstraints{Float64}}}}},ScaledHotVector{Float64}})   # time: 5.3838816
    isdefined(FrankWolfe, Symbol("#push_trajectory!#9")) && Base.precompile(Tuple{getfield(FrankWolfe, Symbol("#push_trajectory!#9")),NamedTuple{(:t, :primal, :dual, :dual_gap, :time, :x, :v, :gamma), Tuple{Int64, Float64, Float64, Float64, Float64, ScaledHotVector{Float64}, Vector{Float64}, Float64}}})   # time: 0.011832673
    Base.precompile(Tuple{typeof(fast_dot),Vector{Float64},SparseVector{Float64, Int64}})   # time: 0.004435971
    Base.precompile(Tuple{typeof(line_search_wrapper),Shortstep,Int64,Function,Function,Vector{Float64},Vector{Float64},SparseVector{Float64, Int64},Float64,Int64,Int64,Float64,Int64,Float64})   # time: 0.002404597
    Base.precompile(Tuple{typeof(line_search_wrapper),Shortstep,Int64,Function,Function,ScaledHotVector{Float64},Vector{Float64},SparseVector{Float64, Int64},Float64,Int64,Int64,Float64,Int64,Float64})   # time: 0.002383438
    Base.precompile(Tuple{Core.kwftype(typeof(print_callback)),NamedTuple{(:print_header,), Tuple{Bool}},typeof(print_callback),Vector{String},String})   # time: 0.00103615
end
