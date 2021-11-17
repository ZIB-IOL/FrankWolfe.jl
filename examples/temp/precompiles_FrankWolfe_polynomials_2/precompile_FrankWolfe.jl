function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{Core.kwftype(typeof(lp_separation_oracle)),NamedTuple{(:inplace_loop, :force_fw_step), Tuple{Bool, Bool}},typeof(lp_separation_oracle),LpNormLMO{Float64, 1},ActiveSet{ScaledHotVector{Float64}, Float64, SparseVector{Float64, Int64}},SparseVector{Float64, Int64},Float64,Float64})   # time: 0.15227416
    Base.precompile(Tuple{Core.kwftype(typeof(adaptive_step_size)),NamedTuple{(:gamma_max, :upgrade_accuracy), Tuple{Float64, Bool}},typeof(adaptive_step_size),Function,Function,SparseVector{Float64, Int64},SparseVector{Float64, Int64},SparseVector{Float64, Int64},Nothing})   # time: 0.007469856
    Base.precompile(Tuple{Core.kwftype(typeof(print_callback)),NamedTuple{(:print_header,), Tuple{Bool}},typeof(print_callback),NTuple{9, String},String})   # time: 0.005911772
    isdefined(FrankWolfe, Symbol("#84#86")) && Base.precompile(Tuple{getfield(FrankWolfe, Symbol("#84#86")),ScaledHotVector{Float64}})   # time: 0.003539617
    Base.precompile(Tuple{typeof(print_callback),Tuple{String, String, Float64, Float64, Float64, Float64, Float64, Int64, Int64},String})   # time: 0.00328763
    Base.precompile(Tuple{typeof(line_search_wrapper),Adaptive,Int64,Function,Function,SparseVector{Float64, Int64},SparseVector{Float64, Int64},SparseVector{Float64, Int64},Float64,Float64,Int64,Float64,Int64,Float64})   # time: 0.002189942
    Base.precompile(Tuple{Core.kwftype(typeof(adaptive_step_size)),NamedTuple{(:gamma_max, :upgrade_accuracy), Tuple{Float64, Bool}},typeof(adaptive_step_size),Function,Function,SparseVector{Float64, Int64},SparseVector{Float64, Int64},SparseVector{Float64, Int64},Float64})   # time: 0.001203851
end
