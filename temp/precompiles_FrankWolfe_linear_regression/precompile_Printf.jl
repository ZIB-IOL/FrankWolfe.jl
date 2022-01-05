function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(format),Base.TTY,Format{Base.CodeUnits{UInt8, String}, Tuple{Spec{Val{'s'}}, Spec{Val{'s'}}, Spec{Val{'e'}}, Spec{Val{'e'}}, Spec{Val{'e'}}, Spec{Val{'e'}}, Spec{Val{'e'}}, Spec{Val{'i'}}}},String,String,Vararg{Any}})   # time: 0.3794015
    Base.precompile(Tuple{typeof(format),Base.TTY,Format{Base.CodeUnits{UInt8, String}, NTuple{8, Spec{Val{'s'}}}},String,String,Vararg{String}})   # time: 0.06983923
    Base.precompile(Tuple{typeof(format),Vector{UInt8},Int64,Format{Base.CodeUnits{UInt8, String}, Tuple{Spec{Val{'s'}}, Spec{Val{'s'}}, Spec{Val{'e'}}, Spec{Val{'e'}}, Spec{Val{'e'}}, Spec{Val{'e'}}, Spec{Val{'e'}}, Spec{Val{'i'}}}},String,Vararg{Any}})   # time: 0.029622564
    Base.precompile(Tuple{typeof(format),Vector{UInt8},Int64,Format{Base.CodeUnits{UInt8, String}, NTuple{8, Spec{Val{'s'}}}},String,Vararg{String}})   # time: 0.023723973
    Base.precompile(Tuple{typeof(format),Base.TTY,Format{Base.CodeUnits{UInt8, String}, Tuple{Spec{Val{'s'}}}},String})   # time: 0.018162861
    Base.precompile(Tuple{typeof(fmt),Vector{UInt8},Int64,Int64,Spec{Val{'i'}}})   # time: 0.016703684
    Base.precompile(Tuple{typeof(fmt),Vector{UInt8},Int64,Float64,Spec{Val{'e'}}})   # time: 0.011670832
    Base.precompile(Tuple{typeof(plength),Spec{Val{'i'}},Int64})   # time: 0.001949178
    Base.precompile(Tuple{Type{Format},Base.CodeUnits{UInt8, String},Vector{UnitRange{Int64}},NTuple{8, Spec{Val{'s'}}}})   # time: 0.001401631
end
