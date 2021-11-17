function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(haskey),NamedTuple{(:gamma_max,), Tuple{Float64}},Symbol})   # time: 0.04481033
    Base.precompile(Tuple{typeof(getindex),Dict{Int32, Symbol},Int32})   # time: 0.014221558
    Base.precompile(Tuple{typeof(axes),UnitRange{Int64}})   # time: 0.002796745
    Base.precompile(Tuple{typeof(/),Float64,Int64})   # time: 0.001330667
    Base.precompile(Tuple{typeof(/),UInt64,Float64})   # time: 0.001180021
end
