function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(iterate),Zip{Tuple{UnitRange{Int64}, StepRange{Int64, Int64}}}})   # time: 0.003385057
    Base.precompile(Tuple{typeof(iterate),Zip{Tuple{UnitRange{Int64}, StepRange{Int64, Int64}}},Tuple{Int64, Int64}})   # time: 0.002832821
end
