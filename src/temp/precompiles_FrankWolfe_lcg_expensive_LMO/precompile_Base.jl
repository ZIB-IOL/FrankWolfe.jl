function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(getindex),Dict{Int32, Symbol},Int32})   # time: 0.015356711
    Base.precompile(Tuple{typeof(string),Int64})   # time: 0.015276042
    Base.precompile(Tuple{typeof(min),Float64,BigFloat})   # time: 0.002656827
    Base.precompile(Tuple{typeof(mod),Int64,Float64})   # time: 0.002639572
    Base.precompile(Tuple{typeof(/),UInt64,Float64})   # time: 0.001764116
    Base.precompile(Tuple{typeof(max),BigFloat,Float64})   # time: 0.001052433
end
