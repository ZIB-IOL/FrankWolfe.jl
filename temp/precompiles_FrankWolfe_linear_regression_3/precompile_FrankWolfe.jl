function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{Core.kwftype(typeof(frank_wolfe)),NamedTuple{(:verbose, :line_search, :max_iteration, :print_iter, :trajectory), Tuple{Bool, Adaptive{Float64, Int64}, Int64, Float64, Bool}},typeof(frank_wolfe),Function,Function,LpNormLMO{Float64, 2},Vector{Float64}})   # time: 0.23070359
    Base.precompile(Tuple{Core.kwftype(typeof(Type)),NamedTuple{(:L_est,), Tuple{Float64}},Type{Adaptive}})   # time: 0.011842306
    Base.precompile(Tuple{Core.kwftype(typeof(print_callback)),NamedTuple{(:print_header,), Tuple{Bool}},typeof(print_callback),Vector{String},String})   # time: 0.00254834
end
