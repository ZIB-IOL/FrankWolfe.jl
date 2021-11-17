function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(format),Base.TTY,Format{Base.CodeUnits{UInt8, String}, NTuple{9, Spec{Val{'s'}}}},String,String,Vararg{String, N} where N})   # time: 0.06700621
    Base.precompile(Tuple{typeof(format),Base.TTY,Format{Base.CodeUnits{UInt8, String}, Tuple{Spec{Val{'s'}}, Spec{Val{'s'}}, Spec{Val{'e'}}, Spec{Val{'e'}}, Spec{Val{'e'}}, Spec{Val{'e'}}, Spec{Val{'e'}}, Spec{Val{'i'}}, Spec{Val{'i'}}}},String,String,Vararg{Any, N} where N})   # time: 0.032841895
    Base.precompile(Tuple{typeof(format),Vector{UInt8},Int64,Format{Base.CodeUnits{UInt8, String}, NTuple{9, Spec{Val{'s'}}}},String,Vararg{String, N} where N})   # time: 0.031879354
    Base.precompile(Tuple{typeof(format),Vector{UInt8},Int64,Format{Base.CodeUnits{UInt8, String}, Tuple{Spec{Val{'s'}}, Spec{Val{'s'}}, Spec{Val{'e'}}, Spec{Val{'e'}}, Spec{Val{'e'}}, Spec{Val{'e'}}, Spec{Val{'e'}}, Spec{Val{'i'}}, Spec{Val{'i'}}}},String,Vararg{Any, N} where N})   # time: 0.015802898
    Base.precompile(Tuple{typeof(computelen),Vector{UnitRange{Int64}},Tuple{Spec{Val{'s'}}, Spec{Val{'s'}}, Spec{Val{'e'}}, Spec{Val{'e'}}, Spec{Val{'e'}}, Spec{Val{'e'}}, Spec{Val{'e'}}, Spec{Val{'i'}}, Spec{Val{'i'}}},Tuple{String, String, Float64, Float64, Float64, Float64, Float64, Int64, Int64}})   # time: 0.005368753
end
