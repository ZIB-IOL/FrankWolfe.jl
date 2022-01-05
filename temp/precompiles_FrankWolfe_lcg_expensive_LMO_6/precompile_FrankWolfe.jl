function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    isdefined(FrankWolfe, Symbol("#push_trajectory!#33")) && Base.precompile(Tuple{getfield(FrankWolfe, Symbol("#push_trajectory!#33")),NamedTuple{(:t, :primal, :dual, :dual_gap, :time, :x, :active_set_length, :non_simplex_iter), Tuple{Int64, Float64, Float64, Float64, Float64, SparseArrays.SparseMatrixCSC{Float64, Int64}, Int64, Int64}}})   # time: 0.025683403
    Base.precompile(Tuple{Core.kwftype(typeof(print_callback)),NamedTuple{(:print_header,), Tuple{Bool}},typeof(print_callback),NTuple{9, String},String})   # time: 0.009511338
    Base.precompile(Tuple{Core.kwftype(typeof(lp_separation_oracle)),NamedTuple{(:inplace_loop, :force_fw_step), Tuple{Bool, Bool}},typeof(lp_separation_oracle),BirkhoffPolytopeLMO,ActiveSet{SparseArrays.SparseMatrixCSC{Float64, Int64}, Float64, SparseArrays.SparseMatrixCSC{Float64, Int64}},SparseArrays.SparseMatrixCSC{Float64, Int64},Float64,Float64})   # time: 0.001432573
end
