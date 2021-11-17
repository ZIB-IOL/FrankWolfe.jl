function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(ht_keyindex2!),Dict{K, V} where {K<:Tuple{Any}, V},Tuple{Any}})   # time: 0.07378435
    Base.precompile(Tuple{typeof(ht_keyindex2!),Dict{K, V} where {K<:Tuple, V},Tuple})   # time: 0.038281854
    Base.precompile(Tuple{typeof(ht_keyindex),Dict{Tuple{DataType, DataType}, Dict{Int64, Int64}},Tuple{DataType, DataType}})   # time: 0.03073506
    Base.precompile(Tuple{typeof(setindex!),Dict{Tuple{DataType, DataType}, Dict{Int64, Int64}},Dict{Int64, Int64},Tuple{DataType, DataType}})   # time: 0.026574062
    Base.precompile(Tuple{typeof(getindex),Dict{Int32, Symbol},Int32})   # time: 0.008258173
    Base.precompile(Tuple{typeof(string),Int64})   # time: 0.006396733
    Base.precompile(Tuple{typeof(setindex!),Vector{_A} where _A,Vector{_A} where _A,Int64})   # time: 0.005097147
    Base.precompile(Tuple{Type{Pair},Any,Tuple{Int64, Any}})   # time: 0.004503018
    Base.precompile(Tuple{typeof(_setindex!),Dict{K, V} where {K<:Tuple{Any}, V},Any,Tuple{Any},Int64})   # time: 0.00365418
    Base.precompile(Tuple{typeof(setindex!),Dict{Any, DataType},DataType,Tuple{DataType}})   # time: 0.003529861
    Base.precompile(Tuple{typeof(_setindex!),Dict{K, V} where {K<:Tuple, V},Any,Tuple,Int64})   # time: 0.003524833
    Base.precompile(Tuple{typeof(push!),Vector{Union{Nothing, DataType}},Nothing})   # time: 0.003411483
    Base.precompile(Tuple{typeof(setindex!),Vector{Tuple{DataType}},Tuple{Any},Union{Integer, CartesianIndex}})   # time: 0.003293032
    Base.precompile(Tuple{typeof(setindex!),Vector{Tuple{DataType, DataType}},Tuple{Any, Any},Union{Integer, CartesianIndex}})   # time: 0.00298993
    Base.precompile(Tuple{typeof(setindex!),Vector{Int32},Int64,Union{Integer, CartesianIndex}})   # time: 0.002662657
    Base.precompile(Tuple{typeof(setindex!),Vector{_A} where _A,Float64,Int64})   # time: 0.002643521
    Base.precompile(Tuple{typeof(copyto!),Vector{Int64},ValueIterator{Dict{Int64, Int64}}})   # time: 0.002414585
    Base.precompile(Tuple{typeof(convert),Type{Tuple{DataType, DataType}},Core.Tuple{Core.DataType, Core.Type{S} where S<:MathOptInterface.AbstractSet}})   # time: 0.002371355
    Base.precompile(Tuple{Type{Dict{Int64, Int64}},Vector{Pair{Int64, Int64}}})   # time: 0.002199899
    Base.precompile(Tuple{typeof(get),Dict{Any, DataType},Tuple{DataType},Nothing})   # time: 0.00218027
    Base.precompile(Tuple{typeof(convert),Type{Tuple{DataType}},Core.Tuple{Core.Type{S} where S<:MathOptInterface.AbstractSet}})   # time: 0.001845247
    Base.precompile(Tuple{typeof(copyto!),Vector{Int64},KeySet{Int64, Dict{Int64, Int64}}})   # time: 0.001841406
    Base.precompile(Tuple{typeof(axes),UnitRange{Int64}})   # time: 0.001699841
    Base.precompile(Tuple{typeof(hash),Tuple{DataType}})   # time: 0.001484803
    Base.precompile(Tuple{typeof(isequal),Tuple{Any},Core.Tuple{Core.Type{S} where S<:MathOptInterface.AbstractSet}})   # time: 0.00147286
    Base.precompile(Tuple{typeof(setindex_widen_up_to),Vector{_A} where _A,Vector{_A} where _A,Int64})   # time: 0.001391634
    Base.precompile(Tuple{typeof(mod),Int64,Float64})   # time: 0.001355804
    Base.precompile(Tuple{typeof(push!),Vector{Tuple{DataType}},Core.Tuple{Core.Type{S} where S<:MathOptInterface.AbstractSet}})   # time: 0.001353504
    Base.precompile(Tuple{typeof(isequal),Tuple{Any, Any},Core.Tuple{Core.DataType, Core.Type{S} where S<:MathOptInterface.AbstractSet}})   # time: 0.001155646
end
