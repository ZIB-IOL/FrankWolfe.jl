function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(format),Base.TTY,Format{Base.CodeUnits{UInt8, String}, NTuple{9, Spec{Val{'s'}}}},String,String,Vararg{String, N} where N})   # time: 0.04980747
    Base.precompile(Tuple{typeof(format),Base.TTY,Format{Base.CodeUnits{UInt8, String}, Tuple{Spec{Val{'s'}}, Spec{Val{'s'}}, Spec{Val{'e'}}, Spec{Val{'e'}}, Spec{Val{'e'}}, Spec{Val{'e'}}, Spec{Val{'e'}}, Spec{Val{'i'}}, Spec{Val{'i'}}}},String,String,Vararg{Any, N} where N})   # time: 0.046805277
    Base.precompile(Tuple{typeof(format),Vector{UInt8},Int64,Format{Base.CodeUnits{UInt8, String}, NTuple{9, Spec{Val{'s'}}}},String,Vararg{String, N} where N})   # time: 0.02459085
    Base.precompile(Tuple{typeof(format),Vector{UInt8},Int64,Format{Base.CodeUnits{UInt8, String}, Tuple{Spec{Val{'s'}}, Spec{Val{'s'}}, Spec{Val{'e'}}, Spec{Val{'e'}}, Spec{Val{'e'}}, Spec{Val{'e'}}, Spec{Val{'e'}}, Spec{Val{'i'}}, Spec{Val{'i'}}}},String,Vararg{Any, N} where N})   # time: 0.016218293
    Base.precompile(Tuple{typeof(computelen),Vector{UnitRange{Int64}},Tuple{Spec{Val{'s'}}, Spec{Val{'s'}}, Spec{Val{'e'}}, Spec{Val{'e'}}, Spec{Val{'e'}}, Spec{Val{'e'}}, Spec{Val{'e'}}, Spec{Val{'i'}}, Spec{Val{'i'}}},Tuple{String, String, Float64, Float64, Float64, Float64, Float64, Int64, Int64}})   # time: 0.005862892
end
