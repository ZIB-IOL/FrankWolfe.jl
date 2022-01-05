function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{Core.kwftype(typeof(frank_wolfe)),NamedTuple{(:max_iteration, :line_search, :print_iter, :memory_mode, :verbose, :trajectory), Tuple{Int64, Shortstep{Float64}, Int64, InplaceEmphasis, Bool, Bool}},typeof(frank_wolfe),Function,Function,ScaledBoundLInfNormBall{Float64, Vector{Float64}, Vector{Float64}},Vector{Float64}})   # time: 0.046453062
    Base.precompile(Tuple{Core.kwftype(typeof(print_callback)),NamedTuple{(:print_header,), Tuple{Bool}},typeof(print_callback),Vector{String},String})   # time: 0.001838422
end
