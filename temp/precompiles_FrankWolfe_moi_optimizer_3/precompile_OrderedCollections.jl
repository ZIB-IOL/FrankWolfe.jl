function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(hashindex),Tuple{Core.Type{<:MathOptInterface.AbstractScalarFunction}},Int64})   # time: 0.004209581
    Base.precompile(Tuple{typeof(hashindex),Tuple{Core.Type{<:MathOptInterface.AbstractSet}},Int64})   # time: 0.003396842
    Base.precompile(Tuple{typeof(hashindex),Tuple{Core.Type{<:MathOptInterface.AbstractFunction}, Core.Type{<:MathOptInterface.AbstractSet}},Int64})   # time: 0.002076643
    Base.precompile(Tuple{typeof(hashindex),Tuple{Core.Type{<:MathOptInterface.AbstractScalarSet}},Int64})   # time: 0.001972861
    Base.precompile(Tuple{typeof(hashindex),Tuple{Core.Type{<:MathOptInterface.AbstractVectorSet}},Int64})   # time: 0.001892055
    Base.precompile(Tuple{typeof(hashindex),Tuple{Core.DataType, Core.Type{<:MathOptInterface.AbstractSet}},Int64})   # time: 0.001710399
    Base.precompile(Tuple{typeof(hashindex),Tuple{Core.DataType, Core.Type{<:MathOptInterface.AbstractScalarSet}},Int64})   # time: 0.001658576
    Base.precompile(Tuple{typeof(hashindex),Tuple{DataType},Int64})   # time: 0.001526229
end
