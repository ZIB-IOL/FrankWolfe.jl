function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{Core.kwftype(typeof(print_callback)),NamedTuple{(:print_header,), Tuple{Bool}},typeof(print_callback),NTuple{9, String},String})   # time: 0.005392282
    Base.precompile(Tuple{Core.kwftype(typeof(lp_separation_oracle)),NamedTuple{(:inplace_loop, :force_fw_step), Tuple{Bool, Bool}},typeof(lp_separation_oracle),BirkhoffPolytopeLMO,ActiveSet{SparseArrays.SparseMatrixCSC{Float64, Int64}, Float64, SparseArrays.SparseMatrixCSC{Float64, Int64}},SparseArrays.SparseMatrixCSC{Float64, Int64},Float64,Float64})   # time: 0.001709011
end
