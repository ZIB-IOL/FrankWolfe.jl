function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(iterate),Zip{Tuple{UnitRange{Int64}, StepRange{Int64, Int64}}}})   # time: 0.002069601
    Base.precompile(Tuple{typeof(iterate),Zip{Tuple{UnitRange{Int64}, StepRange{Int64, Int64}}},Tuple{Int64, Int64}})   # time: 0.001400756
end
