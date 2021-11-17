function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(format),Base.TTY,Format{Base.CodeUnits{UInt8, String}, Tuple{Spec{Val{'s'}}, Spec{Val{'s'}}, Spec{Val{'e'}}, Spec{Val{'e'}}, Spec{Val{'e'}}, Spec{Val{'e'}}, Spec{Val{'e'}}}},String,String,Vararg{Any, N} where N})   # time: 0.26688227
    Base.precompile(Tuple{typeof(format),Base.TTY,Format{Base.CodeUnits{UInt8, String}, NTuple{7, Spec{Val{'s'}}}},String,String,Vararg{String, N} where N})   # time: 0.042391255
    Base.precompile(Tuple{typeof(fmt),Vector{UInt8},Int64,Float64,Spec{Val{'e'}}})   # time: 0.03637557
    Base.precompile(Tuple{typeof(format),Vector{UInt8},Int64,Format{Base.CodeUnits{UInt8, String}, Tuple{Spec{Val{'s'}}, Spec{Val{'s'}}, Spec{Val{'e'}}, Spec{Val{'e'}}, Spec{Val{'e'}}, Spec{Val{'e'}}, Spec{Val{'e'}}}},String,Vararg{Any, N} where N})   # time: 0.03627487
    Base.precompile(Tuple{typeof(format),Base.TTY,Format{Base.CodeUnits{UInt8, String}, Tuple{Spec{Val{'s'}}}},String})   # time: 0.024729434
    Base.precompile(Tuple{typeof(format),Vector{UInt8},Int64,Format{Base.CodeUnits{UInt8, String}, NTuple{7, Spec{Val{'s'}}}},String,Vararg{String, N} where N})   # time: 0.02118748
end
