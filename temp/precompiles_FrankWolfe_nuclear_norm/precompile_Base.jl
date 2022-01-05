function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(getindex),Dict{Int32, Symbol},Int32})   # time: 0.014788095
    Base.precompile(Tuple{typeof(string),Int64})   # time: 0.012464629
    Base.precompile(Tuple{typeof(mod),Int64,Float64})   # time: 0.003570474
    Base.precompile(Tuple{typeof(max),Float64,Int64})   # time: 0.00258816
    Base.precompile(Tuple{typeof(/),Float64,Int64})   # time: 0.00159251
    Base.precompile(Tuple{typeof(zeros),Type{Int64},Int64})   # time: 0.001567858
    Base.precompile(Tuple{typeof(flush),TTY})   # time: 0.001524558
    Base.precompile(Tuple{typeof(/),UInt64,Float64})   # time: 0.001462051
    Base.precompile(Tuple{typeof(iterate),Tuple{String, DataType, String, Type{SubArray{_A, 1, _B, Tuple{UnitRange{Int64}}, true}} where {_A, _B}, String, DataType, String, Any, String, Type{SubArray{Union{}, 1, _B, Tuple{UnitRange{Int64}}, true}} where _B, String, Type{Vector{_A}} where _A, String},Int64})   # time: 0.001319292
    Base.precompile(Tuple{typeof(iterate),Tuple{String, DataType, String, Type{SubArray{_A, 1, _B, Tuple{UnitRange{Int64}}, true}} where {_A, _B}, String, DataType, String, Any, String, Type{SubArray{Union{}, 1, _B, Tuple{UnitRange{Int64}}, true}} where _B, String, Type{Vector{_A}} where _A, String}})   # time: 0.001265373
end
