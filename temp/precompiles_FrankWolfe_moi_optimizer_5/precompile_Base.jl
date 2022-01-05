function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(getindex),Dict{Int32, Symbol},Int32})   # time: 0.015500107
    Base.precompile(Tuple{typeof(string),Int64})   # time: 0.012859136
    Base.precompile(Tuple{typeof(mod),Int64,Float64})   # time: 0.004263913
    Base.precompile(Tuple{typeof(/),UInt64,Float64})   # time: 0.001630571
    Base.precompile(Tuple{typeof(iterate),UnitRange{Int64}})   # time: 0.001422727
    Base.precompile(Tuple{typeof(axes),UnitRange{Int64}})   # time: 0.001292401
end
