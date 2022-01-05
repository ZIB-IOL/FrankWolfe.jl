function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{Core.kwftype(typeof(frank_wolfe)),NamedTuple{(:max_iteration, :line_search, :print_iter, :memory_mode, :verbose, :trajectory), Tuple{Int64, Shortstep{Float64}, Int64, InplaceEmphasis, Bool, Bool}},typeof(frank_wolfe),Function,Function,ScaledBoundL1NormBall{Float64, Vector{Float64}, Vector{Float64}},Vector{Float64}})   # time: 0.19644113
    isdefined(FrankWolfe, Symbol("#push_trajectory!#33")) && Base.precompile(Tuple{getfield(FrankWolfe, Symbol("#push_trajectory!#33")),NamedTuple{(:t, :primal, :dual, :dual_gap, :time, :x, :v, :gamma), Tuple{Int64, Float64, Float64, Float64, Float64, Vector{Float64}, Vector{Float64}, Float64}}})   # time: 0.07687861
    Base.precompile(Tuple{Type{Shortstep},Float64})   # time: 0.037804108
    Base.precompile(Tuple{Core.kwftype(typeof(print_callback)),NamedTuple{(:print_header,), Tuple{Bool}},typeof(print_callback),Vector{String},String})   # time: 0.001267822
end
