function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    isdefined(FrankWolfe, Symbol("#push_trajectory!#9")) && Base.precompile(Tuple{getfield(FrankWolfe, Symbol("#push_trajectory!#9")),NamedTuple{(:t, :primal, :dual, :dual_gap, :time, :x, :v, :active_set_length, :non_simplex_iter), Tuple{Int64, Float64, Float64, Float64, Float64, SparseVector{Float64, Int64}, ScaledHotVector{Float64}, Int64, Int64}}})   # time: 0.010195186
    isdefined(FrankWolfe, Symbol("#push_trajectory!#9")) && Base.precompile(Tuple{getfield(FrankWolfe, Symbol("#push_trajectory!#9")),NamedTuple{(:t, :primal, :dual, :dual_gap, :time, :x), Tuple{Int64, Float64, Float64, Float64, Float64, Vector{Float64}}}})   # time: 0.006906341
    Base.precompile(Tuple{typeof(fast_dot),Any,Vector{Float64}})   # time: 0.003832476
    Base.precompile(Tuple{typeof(fast_dot),Any,Vector{ComplexF64}})   # time: 0.00375332
    Base.precompile(Tuple{Core.kwftype(typeof(print_callback)),NamedTuple{(:print_header,), Tuple{Bool}},typeof(print_callback),NTuple{9, String},String})   # time: 0.003582782
    isdefined(FrankWolfe, Symbol("#reduced_grad!#72")) && Base.precompile(Tuple{getfield(FrankWolfe, Symbol("#reduced_grad!#72")),Vector{Float64},Vector{Float64}})   # time: 0.001215681
end
