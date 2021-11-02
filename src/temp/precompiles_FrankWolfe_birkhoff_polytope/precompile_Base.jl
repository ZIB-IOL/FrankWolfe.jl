function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(getindex),Dict{Int32, Symbol},Int32})   # time: 0.016960988
    Base.precompile(Tuple{typeof(axes),UnitRange{Int64}})   # time: 0.003092919
    Base.precompile(Tuple{typeof(/),UInt64,Float64})   # time: 0.001936905
    Base.precompile(Tuple{typeof(/),Float64,Int64})   # time: 0.001394939
    Base.precompile(Tuple{Colon,Int64,Int64})   # time: 0.00130938
    Base.precompile(Tuple{typeof(indexed_iterate),Tuple{Union{Float64, Int64, BigFloat}, Union{Nothing, Float64}},Int64})   # time: 0.00122655
    Base.precompile(Tuple{typeof(indexed_iterate),Tuple{Union{Float64, Int64, BigFloat}, Float64},Int64})   # time: 0.00100655
end
