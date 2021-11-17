function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    isdefined(SparseArrays.HigherOrderFns, Symbol("#3#4")) && Base.precompile(Tuple{getfield(SparseArrays.HigherOrderFns, Symbol("#3#4")),Float64,Float64,Float64})   # time: 0.002834886
    isdefined(SparseArrays.HigherOrderFns, Symbol("#3#4")) && Base.precompile(Tuple{getfield(SparseArrays.HigherOrderFns, Symbol("#3#4")),Float64,Float64,Float64})   # time: 0.00239594
    Base.precompile(Tuple{typeof(_isemptycol_all),Tuple{Int64, Int64, Int64},Tuple{Int64, Int64, Int64}})   # time: 0.001280103
end
