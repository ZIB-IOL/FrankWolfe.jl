function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(getindex),Dict{Int32, Symbol},Int32})   # time: 0.020800808
    Base.precompile(Tuple{typeof(axes),UnitRange{Int64}})   # time: 0.003647568
    Base.precompile(Tuple{typeof(/),Float64,Int64})   # time: 0.001965455
    Base.precompile(Tuple{typeof(/),UInt64,Float64})   # time: 0.001626236
    Base.precompile(Tuple{Colon,Int64,Int64})   # time: 0.001577627
    Base.precompile(Tuple{typeof(to_shape),Tuple{OneTo{Int64}}})   # time: 0.001252101
end
