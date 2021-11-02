function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(show),IOBuffer,Type})   # time: 0.12601502
    Base.precompile(Tuple{typeof(getindex),Dict{Int32, Symbol},Int32})   # time: 0.018862138
    Base.precompile(Tuple{typeof(+),Any,Float64,Float64,Float64})   # time: 0.012977735
    Base.precompile(Tuple{typeof(axes),UnitRange{Int64}})   # time: 0.003076291
    Base.precompile(Tuple{typeof(/),UInt64,Float64})   # time: 0.001728776
    Base.precompile(Tuple{typeof(/),Float64,Int64})   # time: 0.001439497
    Base.precompile(Tuple{typeof(*),Int64,Float64})   # time: 0.00134095
    Base.precompile(Tuple{typeof(+),Int64,Float64})   # time: 0.001119832
    Base.precompile(Tuple{Colon,Int64,Int64})   # time: 0.001031696
end
