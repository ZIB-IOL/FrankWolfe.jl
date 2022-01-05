function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(setindex!),Dict{Any, Type},Any,Tuple{Core.DataType, Core.Type{<:MathOptInterface.AbstractVectorSet}}})   # time: 0.015220163
    Base.precompile(Tuple{typeof(getindex),Dict{Int32, Symbol},Int32})   # time: 0.007933037
    Base.precompile(Tuple{typeof(string),String,Core.Type{<:MathOptInterface.AbstractSet},String})   # time: 0.006863341
    Base.precompile(Tuple{typeof(string),String,Core.Type{<:MathOptInterface.AbstractScalarSet},String})   # time: 0.006838334
    Base.precompile(Tuple{typeof(get),Dict{Any, Type},Tuple{Core.DataType, Core.Type{<:MathOptInterface.AbstractVectorSet}},Nothing})   # time: 0.005141828
    Base.precompile(Tuple{typeof(string),Int64})   # time: 0.004990693
    Base.precompile(Tuple{typeof(setindex!),Dict{Any, Type},Any,Tuple{DataType, DataType}})   # time: 0.003562659
    Base.precompile(Tuple{typeof(setindex!),Dict{Any, Type},Type,Tuple{Core.Type{<:MathOptInterface.AbstractVectorSet}}})   # time: 0.003121329
    Base.precompile(Tuple{typeof(setindex!),Dict{Any, Type},Any,Tuple{Core.DataType, Core.Type{<:MathOptInterface.AbstractScalarSet}}})   # time: 0.002973127
    Base.precompile(Tuple{typeof(get),Dict{Any, Type},Tuple{Core.Type{<:MathOptInterface.AbstractScalarFunction}},Nothing})   # time: 0.002789374
    Base.precompile(Tuple{typeof(setindex!),Dict{Any, Type},Type,Tuple{Core.Type{<:MathOptInterface.AbstractSet}}})   # time: 0.002777992
    Base.precompile(Tuple{typeof(get),Dict{Any, Type},Tuple{Core.Type{<:MathOptInterface.AbstractFunction}, Core.Type{<:MathOptInterface.AbstractSet}},Nothing})   # time: 0.00276193
    Base.precompile(Tuple{typeof(setindex!),Dict{Any, Type},Type,Tuple{Core.Type{<:MathOptInterface.AbstractScalarFunction}}})   # time: 0.002759836
    Base.precompile(Tuple{typeof(vect),Int64,Int64})   # time: 0.002743211
    Base.precompile(Tuple{typeof(setindex!),Dict{Any, Type},Type,Tuple{DataType}})   # time: 0.00273937
    Base.precompile(Tuple{typeof(setindex!),Dict{Any, Type},Any,Tuple{Core.Type{<:MathOptInterface.AbstractFunction}, Core.Type{<:MathOptInterface.AbstractSet}}})   # time: 0.002658075
    Base.precompile(Tuple{typeof(get),Dict{Any, Type},Tuple{Core.Type{<:MathOptInterface.AbstractScalarSet}},Nothing})   # time: 0.002632866
    Base.precompile(Tuple{typeof(setindex!),Dict{Any, Type},Type,Tuple{Core.Type{<:MathOptInterface.AbstractScalarSet}}})   # time: 0.002545213
    Base.precompile(Tuple{Type{Dict{Int64, Int64}},Vector{Pair{Int64, Int64}}})   # time: 0.002450156
    Base.precompile(Tuple{typeof(get),Dict{Any, Type},Tuple{Core.DataType, Core.Type{<:MathOptInterface.AbstractScalarSet}},Nothing})   # time: 0.002083427
    Base.precompile(Tuple{typeof(get),Dict{Any, Type},Tuple{Core.Type{<:MathOptInterface.AbstractVectorSet}},Nothing})   # time: 0.002074837
    Base.precompile(Tuple{typeof(copyto!),Vector{Int64},ValueIterator{Dict{Int64, Int64}}})   # time: 0.002021341
    Base.precompile(Tuple{typeof(get),Dict{Any, Type},Tuple{Core.Type{<:MathOptInterface.AbstractSet}},Nothing})   # time: 0.001891154
    Base.precompile(Tuple{typeof(get),Dict{Any, Type},Tuple{DataType, DataType},Nothing})   # time: 0.001862939
    Base.precompile(Tuple{typeof(get),Dict{Any, Type},Tuple{DataType},Nothing})   # time: 0.001844358
    Base.precompile(Tuple{typeof(push!),Vector{Union{Nothing, Type}},Core.Type{<:MathOptInterface.AbstractScalarSet}})   # time: 0.001816632
    Base.precompile(Tuple{typeof(mod),Int64,Float64})   # time: 0.001708234
    Base.precompile(Tuple{typeof(isequal),Tuple{Core.DataType, Core.Type{<:MathOptInterface.AbstractVectorSet}},Tuple{Core.DataType, Core.Type{<:MathOptInterface.AbstractVectorSet}}})   # time: 0.00159321
    Base.precompile(Tuple{typeof(isequal),Tuple{Core.Type{<:MathOptInterface.AbstractFunction}, Core.Type{<:MathOptInterface.AbstractSet}},Tuple{Core.Type{<:MathOptInterface.AbstractFunction}, Core.Type{<:MathOptInterface.AbstractSet}}})   # time: 0.001521554
    Base.precompile(Tuple{typeof(isequal),Tuple{Core.DataType, Core.Type{<:MathOptInterface.AbstractVectorSet}},Tuple{Type, Type}})   # time: 0.001486143
    Base.precompile(Tuple{typeof(isequal),Tuple{Core.Type{<:MathOptInterface.AbstractFunction}, Core.Type{<:MathOptInterface.AbstractSet}},Tuple{Type, Type}})   # time: 0.001469033
    Base.precompile(Tuple{typeof(push!),Vector{Union{Nothing, Type}},Core.Type{<:MathOptInterface.AbstractVectorSet}})   # time: 0.001311612
    Base.precompile(Tuple{typeof(setindex!),BitVector,Float64,Int64})   # time: 0.001228712
    Base.precompile(Tuple{typeof(isequal),Tuple{Core.Type{<:MathOptInterface.AbstractSet}},Tuple{Core.Type{<:MathOptInterface.AbstractSet}}})   # time: 0.001180522
    Base.precompile(Tuple{typeof(iterate),Base.Vector{<:MathOptInterface.ConstraintIndex}})   # time: 0.001167951
    Base.precompile(Tuple{typeof(isequal),Tuple{Core.Type{<:MathOptInterface.AbstractScalarFunction}},Tuple{Core.Type{<:MathOptInterface.AbstractScalarFunction}}})   # time: 0.001135055
    Base.precompile(Tuple{typeof(isequal),Tuple{Core.Type{<:MathOptInterface.AbstractSet}},Tuple{Type}})   # time: 0.001119412
    Base.precompile(Tuple{typeof(isequal),Tuple{Core.Type{<:MathOptInterface.AbstractScalarFunction}},Tuple{Type}})   # time: 0.001079252
    Base.precompile(Tuple{typeof(isequal),Tuple{Core.DataType, Core.Type{<:MathOptInterface.AbstractScalarSet}},Tuple{Core.DataType, Core.Type{<:MathOptInterface.AbstractScalarSet}}})   # time: 0.001056413
    Base.precompile(Tuple{typeof(setindex!),Dict{Tuple{Type, Type}, Dict{Int64, Int64}},Dict{Int64, Int64},Tuple{DataType, DataType}})   # time: 0.001027222
    Base.precompile(Tuple{typeof(isequal),Tuple{Core.DataType, Core.Type{<:MathOptInterface.AbstractScalarSet}},Tuple{Type, Type}})   # time: 0.001008364
end
