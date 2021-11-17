function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(format),Base.TTY,Format{Base.CodeUnits{UInt8, String}, Tuple{Spec{Val{'s'}}, Spec{Val{'s'}}, Spec{Val{'e'}}, Spec{Val{'e'}}, Spec{Val{'e'}}, Spec{Val{'e'}}, Spec{Val{'e'}}, Spec{Val{'i'}}}},String,String,Vararg{Any, N} where N})   # time: 0.21232748
    Base.precompile(Tuple{typeof(format),Base.TTY,Format{Base.CodeUnits{UInt8, String}, NTuple{8, Spec{Val{'s'}}}},String,String,Vararg{String, N} where N})   # time: 0.031916436
    Base.precompile(Tuple{typeof(format),Vector{UInt8},Int64,Format{Base.CodeUnits{UInt8, String}, Tuple{Spec{Val{'s'}}, Spec{Val{'s'}}, Spec{Val{'e'}}, Spec{Val{'e'}}, Spec{Val{'e'}}, Spec{Val{'e'}}, Spec{Val{'e'}}, Spec{Val{'i'}}}},String,Vararg{Any, N} where N})   # time: 0.020820964
    Base.precompile(Tuple{typeof(format),Vector{UInt8},Int64,Format{Base.CodeUnits{UInt8, String}, NTuple{8, Spec{Val{'s'}}}},String,Vararg{String, N} where N})   # time: 0.015838025
    Base.precompile(Tuple{typeof(fmt),Vector{UInt8},Int64,Int64,Spec{Val{'i'}}})   # time: 0.010257225
    Base.precompile(Tuple{typeof(format),Base.TTY,Format{Base.CodeUnits{UInt8, String}, Tuple{Spec{Val{'s'}}}},String})   # time: 0.008128788
    Base.precompile(Tuple{typeof(fmt),Vector{UInt8},Int64,Float64,Spec{Val{'e'}}})   # time: 0.006434654
    Base.precompile(Tuple{typeof(plength),Spec{Val{'i'}},Int64})   # time: 0.001157809
end
