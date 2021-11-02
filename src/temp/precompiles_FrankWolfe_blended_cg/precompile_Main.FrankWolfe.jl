function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{Core.kwftype(typeof(print_callback)),NamedTuple{(:print_header,), Tuple{Bool}},typeof(print_callback),NTuple{9, String},String})   # time: 0.010016621
    Base.precompile(Tuple{typeof(print_callback),Tuple{String, String, Float64, Float64, Float64, Float64, Float64, Int64, Int64},String})   # time: 0.007750893
    Base.precompile(Tuple{Core.kwftype(typeof(lp_separation_oracle)),NamedTuple{(:inplace_loop, :force_fw_step), Tuple{Bool, Bool}},typeof(lp_separation_oracle),ProbabilitySimplexOracle{Float64},ActiveSet{ScaledHotVector{Float64}, Float64, SparseVector{Float64, Int64}},SparseVector{Float64, Int64},Float64,Float64})   # time: 0.002545288
end
