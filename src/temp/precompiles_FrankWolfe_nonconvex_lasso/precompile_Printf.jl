function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(format),Base.TTY,Format{Base.CodeUnits{UInt8, String}, NTuple{7, Spec{Val{'s'}}}},String,String,Vararg{String, N} where N})   # time: 0.05740513
    Base.precompile(Tuple{typeof(format),Vector{UInt8},Int64,Format{Base.CodeUnits{UInt8, String}, NTuple{7, Spec{Val{'s'}}}},String,Vararg{String, N} where N})   # time: 0.026851466
    Base.precompile(Tuple{Type{Format},Base.CodeUnits{UInt8, String},Vector{UnitRange{Int64}},NTuple{7, Spec{Val{'s'}}}})   # time: 0.001158174
    Base.precompile(Tuple{Type{Format},Base.CodeUnits{UInt8, String},Vector{UnitRange{Int64}},Tuple{Spec{Val{'s'}}, Spec{Val{'s'}}, Spec{Val{'e'}}, Spec{Val{'e'}}, Spec{Val{'e'}}, Spec{Val{'e'}}, Spec{Val{'e'}}}})   # time: 0.001108517
end
