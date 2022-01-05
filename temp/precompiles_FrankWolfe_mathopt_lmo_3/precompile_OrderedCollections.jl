function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(hashindex),Tuple{Core.Type{<:MathOptInterface.AbstractSet}},Int64})   # time: 0.002239591
    Base.precompile(Tuple{typeof(hashindex),Tuple{Core.Type{<:MathOptInterface.AbstractScalarFunction}},Int64})   # time: 0.00219098
    Base.precompile(Tuple{typeof(hashindex),Tuple{Core.Type{<:MathOptInterface.AbstractFunction}, Core.Type{<:MathOptInterface.AbstractSet}},Int64})   # time: 0.001385714
    Base.precompile(Tuple{typeof(hashindex),Tuple{Core.Type{<:MathOptInterface.AbstractScalarSet}},Int64})   # time: 0.001157196
    Base.precompile(Tuple{typeof(hashindex),Tuple{Core.Type{<:MathOptInterface.AbstractVectorSet}},Int64})   # time: 0.001141967
    Base.precompile(Tuple{typeof(hashindex),Tuple{Core.DataType, Core.Type{<:MathOptInterface.AbstractScalarSet}},Int64})   # time: 0.001118783
    Base.precompile(Tuple{typeof(hashindex),Tuple{Core.DataType, Core.Type{<:MathOptInterface.AbstractSet}},Int64})   # time: 0.001101532
end
