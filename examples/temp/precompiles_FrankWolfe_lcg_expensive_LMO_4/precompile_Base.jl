function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(getindex),Dict{Int32, Symbol},Int32})   # time: 0.01392191
    Base.precompile(Tuple{typeof(string),Int64})   # time: 0.01174421
    Base.precompile(Tuple{typeof(max),Int64,Float64})   # time: 0.002068814
    Base.precompile(Tuple{typeof(mod),Int64,Int64})   # time: 0.001370337
    Base.precompile(Tuple{typeof(min),Float64,Int64})   # time: 0.001253706
    Base.precompile(Tuple{typeof(/),UInt64,Float64})   # time: 0.001127992
    Base.precompile(Tuple{typeof(iterate),UnitRange{Int64}})   # time: 0.001036222
    Base.precompile(Tuple{typeof(mod),Int64,Float64})   # time: 0.001021344
end
