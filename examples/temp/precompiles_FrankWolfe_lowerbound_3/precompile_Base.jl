function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(getindex),Dict{Int32, Symbol},Int32})   # time: 0.013955943
    Base.precompile(Tuple{typeof(axes),UnitRange{Int64}})   # time: 0.003443822
    Base.precompile(Tuple{typeof(/),UInt64,Float64})   # time: 0.00151623
    Base.precompile(Tuple{typeof(/),Float64,Int64})   # time: 0.001238473
end
