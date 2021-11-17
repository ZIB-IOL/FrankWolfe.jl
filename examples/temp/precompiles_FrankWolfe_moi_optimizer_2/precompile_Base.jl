function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(getindex),Dict{Int32, Symbol},Int32})   # time: 0.015255329
    Base.precompile(Tuple{typeof(string),Int64})   # time: 0.012845081
    Base.precompile(Tuple{typeof(_array_for),Type{HasShape{1}},UnitRange{Int64},HasShape{1}})   # time: 0.002283635
    Base.precompile(Tuple{typeof(mod),Int64,Float64})   # time: 0.00227414
    Base.precompile(Tuple{typeof(max),Float64,Int64})   # time: 0.001932755
    Base.precompile(Tuple{typeof(/),UInt64,Float64})   # time: 0.001425855
end
