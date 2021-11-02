function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    isdefined(SparseArrays.HigherOrderFns, Symbol("#3#4")) && Base.precompile(Tuple{getfield(SparseArrays.HigherOrderFns, Symbol("#3#4")),Float64,Float64})   # time: 0.007804253
    Base.precompile(Tuple{typeof(_capturescalars),Base.RefValue{typeof(^)},Int64,Base.RefValue{Val{2}}})   # time: 0.002492769
    isdefined(SparseArrays.HigherOrderFns, Symbol("#3#4")) && Base.precompile(Tuple{getfield(SparseArrays.HigherOrderFns, Symbol("#3#4")),Float64,Vararg{Float64, N} where N})   # time: 0.00114267
end
