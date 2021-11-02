function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(getindex),Dict{Int32, Symbol},Int32})   # time: 0.008506791
    Base.precompile(Tuple{typeof(string),Int64})   # time: 0.006600268
    Base.precompile(Tuple{typeof(sum),Generator{Vector{Tuple{Int64, Int64}}, _A} where _A})   # time: 0.004329382
    Base.precompile(Tuple{typeof(mod),Int64,Float64})   # time: 0.001383271
    Base.precompile(Tuple{typeof(zeros),Type{Int64},Int64})   # time: 0.00128389
    Base.precompile(Tuple{typeof(iterate),Tuple{String, DataType, String, Type{SubArray{_A, 1, _B, Tuple{UnitRange{Int64}}, true}} where {_A, _B}, String, DataType, String, Any, String, Type{SubArray{Union{}, 1, _B, Tuple{UnitRange{Int64}}, true}} where _B, String, Type{Vector{_A}} where _A, String}})   # time: 0.001207343
end
