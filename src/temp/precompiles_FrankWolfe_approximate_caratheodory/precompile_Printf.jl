function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(format),Base.TTY,Format{Base.CodeUnits{UInt8, String}, NTuple{7, Spec{Val{'s'}}}},String,String,Vararg{String, N} where N})   # time: 0.041287422
    Base.precompile(Tuple{typeof(format),Vector{UInt8},Int64,Format{Base.CodeUnits{UInt8, String}, NTuple{7, Spec{Val{'s'}}}},String,Vararg{String, N} where N})   # time: 0.020721097
end
