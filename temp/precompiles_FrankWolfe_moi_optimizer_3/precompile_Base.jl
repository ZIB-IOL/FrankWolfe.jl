function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(getindex),Dict{Int32, Symbol},Int32})   # time: 0.064437404
    Base.precompile(Tuple{typeof(setindex!),Dict{Any, Type},Any,Tuple{Core.DataType, Core.Type{<:MathOptInterface.AbstractVectorSet}}})   # time: 0.025321295
    Base.precompile(Tuple{typeof(get),Dict{Any, Type},Tuple{Core.DataType, Core.Type{<:MathOptInterface.AbstractVectorSet}},Nothing})   # time: 0.014061104
    Base.precompile(Tuple{typeof(string),String,Core.Type{<:MathOptInterface.AbstractScalarSet},String})   # time: 0.011196022
    Base.precompile(Tuple{typeof(string),String,Core.Type{<:MathOptInterface.AbstractSet},String})   # time: 0.010534696
    Base.precompile(Tuple{typeof(string),Int64})   # time: 0.008088531
    Base.precompile(Tuple{typeof(vect),Int64,Int64})   # time: 0.007640438
    Base.precompile(Tuple{typeof(setindex!),Dict{Any, Type},Any,Tuple{Core.DataType, Core.Type{<:MathOptInterface.AbstractScalarSet}}})   # time: 0.005586279
    Base.precompile(Tuple{typeof(setindex!),Dict{Any, Type},Any,Tuple{DataType, DataType}})   # time: 0.005498207
    Base.precompile(Tuple{typeof(isequal),Tuple{Core.DataType, Core.Type{<:MathOptInterface.AbstractVectorSet}},Tuple{Type, Type}})   # time: 0.005039148
    Base.precompile(Tuple{typeof(setindex!),Dict{Any, Type},Type,Tuple{Core.Type{<:MathOptInterface.AbstractSet}}})   # time: 0.004951706
    Base.precompile(Tuple{typeof(setindex!),Dict{Any, Type},Type,Tuple{DataType}})   # time: 0.004789325
    Base.precompile(Tuple{typeof(setindex!),Dict{Any, Type},Any,Tuple{Core.Type{<:MathOptInterface.AbstractFunction}, Core.Type{<:MathOptInterface.AbstractSet}}})   # time: 0.004602712
    Base.precompile(Tuple{typeof(setindex!),Dict{Any, Type},Type,Tuple{Core.Type{<:MathOptInterface.AbstractScalarFunction}}})   # time: 0.004458001
    Base.precompile(Tuple{typeof(setindex!),Dict{Any, Type},Type,Tuple{Core.Type{<:MathOptInterface.AbstractVectorSet}}})   # time: 0.004404294
    Base.precompile(Tuple{typeof(setindex!),Dict{Any, Type},Type,Tuple{Core.Type{<:MathOptInterface.AbstractScalarSet}}})   # time: 0.00435897
    Base.precompile(Tuple{typeof(get),Dict{Any, Type},Tuple{Core.Type{<:MathOptInterface.AbstractFunction}, Core.Type{<:MathOptInterface.AbstractSet}},Nothing})   # time: 0.004315736
    Base.precompile(Tuple{typeof(get),Dict{Any, Type},Tuple{Core.Type{<:MathOptInterface.AbstractScalarFunction}},Nothing})   # time: 0.004206502
    Base.precompile(Tuple{typeof(isequal),Tuple{Core.DataType, Core.Type{<:MathOptInterface.AbstractVectorSet}},Tuple{Core.DataType, Core.Type{<:MathOptInterface.AbstractVectorSet}}})   # time: 0.004179546
    Base.precompile(Tuple{typeof(get),Dict{Any, Type},Tuple{Core.Type{<:MathOptInterface.AbstractScalarSet}},Nothing})   # time: 0.003953403
    Base.precompile(Tuple{typeof(push!),Vector{Union{Nothing, Type}},Core.Type{<:MathOptInterface.AbstractScalarSet}})   # time: 0.003494058
    Base.precompile(Tuple{Type{Dict{Int64, Int64}},Vector{Pair{Int64, Int64}}})   # time: 0.003480515
    Base.precompile(Tuple{typeof(get),Dict{Any, Type},Tuple{Core.DataType, Core.Type{<:MathOptInterface.AbstractScalarSet}},Nothing})   # time: 0.003333916
    Base.precompile(Tuple{typeof(get),Dict{Any, Type},Tuple{Core.Type{<:MathOptInterface.AbstractSet}},Nothing})   # time: 0.003298714
    Base.precompile(Tuple{typeof(get),Dict{Any, Type},Tuple{Core.Type{<:MathOptInterface.AbstractVectorSet}},Nothing})   # time: 0.003010975
    Base.precompile(Tuple{typeof(get),Dict{Any, Type},Tuple{DataType},Nothing})   # time: 0.002816815
    Base.precompile(Tuple{typeof(copyto!),Vector{Int64},ValueIterator{Dict{Int64, Int64}}})   # time: 0.002744601
    Base.precompile(Tuple{typeof(mod),Int64,Float64})   # time: 0.002682161
    Base.precompile(Tuple{typeof(get),Dict{Any, Type},Tuple{DataType, DataType},Nothing})   # time: 0.002415161
    Base.precompile(Tuple{typeof(isequal),Tuple{Core.Type{<:MathOptInterface.AbstractFunction}, Core.Type{<:MathOptInterface.AbstractSet}},Tuple{Type, Type}})   # time: 0.002353072
    Base.precompile(Tuple{typeof(isequal),Tuple{Core.Type{<:MathOptInterface.AbstractSet}},Tuple{Core.Type{<:MathOptInterface.AbstractSet}}})   # time: 0.00225802
    Base.precompile(Tuple{typeof(push!),Vector{Union{Nothing, Type}},Core.Type{<:MathOptInterface.AbstractVectorSet}})   # time: 0.002096339
    Base.precompile(Tuple{typeof(isequal),Tuple{Core.Type{<:MathOptInterface.AbstractScalarFunction}},Tuple{Core.Type{<:MathOptInterface.AbstractScalarFunction}}})   # time: 0.002078602
    Base.precompile(Tuple{typeof(isequal),Tuple{Core.Type{<:MathOptInterface.AbstractFunction}, Core.Type{<:MathOptInterface.AbstractSet}},Tuple{Core.Type{<:MathOptInterface.AbstractFunction}, Core.Type{<:MathOptInterface.AbstractSet}}})   # time: 0.002075874
    Base.precompile(Tuple{typeof(isequal),Tuple{Core.Type{<:MathOptInterface.AbstractSet}},Tuple{Type}})   # time: 0.002065121
    Base.precompile(Tuple{typeof(isequal),Tuple{Core.DataType, Core.Type{<:MathOptInterface.AbstractSet}},Tuple{Type, Type}})   # time: 0.001927323
    Base.precompile(Tuple{typeof(setindex!),BitVector,Float64,Int64})   # time: 0.001900152
    Base.precompile(Tuple{typeof(/),UInt64,Float64})   # time: 0.001819142
    Base.precompile(Tuple{typeof(isequal),Tuple{Core.DataType, Core.Type{<:MathOptInterface.AbstractScalarSet}},Tuple{Type, Type}})   # time: 0.001809084
    Base.precompile(Tuple{typeof(iterate),Base.Vector{<:MathOptInterface.ConstraintIndex}})   # time: 0.001741129
    Base.precompile(Tuple{typeof(isequal),Tuple{Core.Type{<:MathOptInterface.AbstractScalarFunction}},Tuple{Type}})   # time: 0.001699645
    Base.precompile(Tuple{typeof(isequal),Tuple{Core.DataType, Core.Type{<:MathOptInterface.AbstractScalarSet}},Tuple{Core.DataType, Core.Type{<:MathOptInterface.AbstractScalarSet}}})   # time: 0.001614926
    Base.precompile(Tuple{typeof(isequal),Tuple{Core.DataType, Core.Type{<:MathOptInterface.AbstractSet}},Tuple{Core.DataType, Core.Type{<:MathOptInterface.AbstractSet}}})   # time: 0.001593484
    Base.precompile(Tuple{typeof(_similar_shape),UnitRange{Int64},HasShape{1}})   # time: 0.001545646
    Base.precompile(Tuple{typeof(setindex!),Dict{Tuple{Type, Type}, Dict{Int64, Int64}},Dict{Int64, Int64},Tuple{DataType, DataType}})   # time: 0.001375721
    Base.precompile(Tuple{typeof(isassigned),Vector{Tuple{Type}},Int64})   # time: 0.001373417
    Base.precompile(Tuple{typeof(push!),Vector{Tuple{Type}},Tuple{Core.Type{<:MathOptInterface.AbstractScalarFunction}}})   # time: 0.001245119
    Base.precompile(Tuple{typeof(max),Float64,Int64})   # time: 0.001143013
    Base.precompile(Tuple{typeof(push!),Vector{Union{Nothing, Type}},Nothing})   # time: 0.001133655
    Base.precompile(Tuple{typeof(fill!),Vector{Int32},Int64})   # time: 0.001076942
    Base.precompile(Tuple{typeof(hash),Tuple{Type}})   # time: 0.001027776
    Base.precompile(Tuple{typeof(push!),Vector{Tuple{Type, Type}},Tuple{Core.Type{<:MathOptInterface.AbstractFunction}, Core.Type{<:MathOptInterface.AbstractSet}}})   # time: 0.001019325
    Base.precompile(Tuple{typeof(push!),Vector{Tuple{Type}},Tuple{Core.Type{<:MathOptInterface.AbstractVectorSet}}})   # time: 0.001019325
    Base.precompile(Tuple{typeof(getindex),Vector{Tuple{Type, Type}},Int32})   # time: 0.001011787
end
