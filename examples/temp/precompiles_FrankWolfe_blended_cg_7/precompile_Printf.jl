function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(format),Base.TTY,Format{Base.CodeUnits{UInt8, String}, NTuple{8, Spec{Val{'s'}}}},String,String,Vararg{String, N} where N})   # time: 0.031410683
    Base.precompile(Tuple{typeof(format),Base.TTY,Format{Base.CodeUnits{UInt8, String}, Tuple{Spec{Val{'s'}}, Spec{Val{'s'}}, Spec{Val{'e'}}, Spec{Val{'e'}}, Spec{Val{'e'}}, Spec{Val{'e'}}, Spec{Val{'e'}}, Spec{Val{'i'}}}},String,String,Vararg{Any, N} where N})   # time: 0.020508563
    Base.precompile(Tuple{typeof(format),Vector{UInt8},Int64,Format{Base.CodeUnits{UInt8, String}, NTuple{8, Spec{Val{'s'}}}},String,Vararg{String, N} where N})   # time: 0.015076175
    Base.precompile(Tuple{typeof(format),Vector{UInt8},Int64,Format{Base.CodeUnits{UInt8, String}, Tuple{Spec{Val{'s'}}, Spec{Val{'s'}}, Spec{Val{'e'}}, Spec{Val{'e'}}, Spec{Val{'e'}}, Spec{Val{'e'}}, Spec{Val{'e'}}, Spec{Val{'i'}}}},String,Vararg{Any, N} where N})   # time: 0.009655468
end
