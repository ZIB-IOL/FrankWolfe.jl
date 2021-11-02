function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(show),IOBuffer,Type})   # time: 0.13670439
    Base.precompile(Tuple{typeof(getindex),Dict{Int32, Symbol},Int32})   # time: 0.014643374
    Base.precompile(Tuple{typeof(string),Int64})   # time: 0.011186659
    Base.precompile(Tuple{typeof(max),Float64,Int64})   # time: 0.002112342
    Base.precompile(Tuple{typeof(mod),Int64,Float64})   # time: 0.002067783
    Base.precompile(Tuple{typeof(/),UInt64,Float64})   # time: 0.00137489
end
