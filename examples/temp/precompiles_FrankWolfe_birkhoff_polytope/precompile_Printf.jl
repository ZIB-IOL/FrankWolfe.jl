function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(format),Base.TTY,Format{Base.CodeUnits{UInt8, String}, Tuple{Spec{Val{'s'}}, Spec{Val{'s'}}, Spec{Val{'e'}}, Spec{Val{'e'}}, Spec{Val{'e'}}, Spec{Val{'e'}}, Spec{Val{'e'}}, Spec{Val{'i'}}, Spec{Val{'i'}}}},String,String,Vararg{Any, N} where N})   # time: 0.21891226
    Base.precompile(Tuple{typeof(format),Base.TTY,Format{Base.CodeUnits{UInt8, String}, NTuple{9, Spec{Val{'s'}}}},String,String,Vararg{String, N} where N})   # time: 0.031642467
    Base.precompile(Tuple{typeof(format),Vector{UInt8},Int64,Format{Base.CodeUnits{UInt8, String}, Tuple{Spec{Val{'s'}}, Spec{Val{'s'}}, Spec{Val{'e'}}, Spec{Val{'e'}}, Spec{Val{'e'}}, Spec{Val{'e'}}, Spec{Val{'e'}}, Spec{Val{'i'}}, Spec{Val{'i'}}}},String,Vararg{Any, N} where N})   # time: 0.021231944
    Base.precompile(Tuple{typeof(format),Vector{UInt8},Int64,Format{Base.CodeUnits{UInt8, String}, NTuple{9, Spec{Val{'s'}}}},String,Vararg{String, N} where N})   # time: 0.01747404
    Base.precompile(Tuple{typeof(fmt),Vector{UInt8},Int64,Int64,Spec{Val{'i'}}})   # time: 0.008753813
    Base.precompile(Tuple{typeof(format),Base.TTY,Format{Base.CodeUnits{UInt8, String}, Tuple{Spec{Val{'s'}}}},String})   # time: 0.007237559
    Base.precompile(Tuple{typeof(fmt),Vector{UInt8},Int64,Float64,Spec{Val{'e'}}})   # time: 0.006093115
    Base.precompile(Tuple{typeof(computelen),Vector{UnitRange{Int64}},Tuple{Spec{Val{'s'}}, Spec{Val{'s'}}, Spec{Val{'e'}}, Spec{Val{'e'}}, Spec{Val{'e'}}, Spec{Val{'e'}}, Spec{Val{'e'}}, Spec{Val{'i'}}, Spec{Val{'i'}}},Tuple{String, String, Float64, Float64, Float64, Float64, Float64, Int64, Int64}})   # time: 0.004652135
end
