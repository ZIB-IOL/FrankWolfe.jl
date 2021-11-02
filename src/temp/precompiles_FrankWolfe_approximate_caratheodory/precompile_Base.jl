function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(getindex),Dict{Int32, Symbol},Int32})   # time: 0.015444248
    Base.precompile(Tuple{typeof(string),Int64})   # time: 0.011794696
    Base.precompile(Tuple{typeof(mod),Int64,Float64})   # time: 0.002564915
    Base.precompile(Tuple{typeof(/),UInt64,Float64})   # time: 0.00130966
    Base.precompile(Tuple{typeof(iterate),UnitRange{Int64}})   # time: 0.00100697
end
