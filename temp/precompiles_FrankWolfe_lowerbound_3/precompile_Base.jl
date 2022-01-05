function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(getindex),Dict{Int32, Symbol},Int32})   # time: 0.009577463
    Base.precompile(Tuple{typeof(min),Int64,Float64})   # time: 0.001518264
    Base.precompile(Tuple{typeof(max),Int64,Float64})   # time: 0.001334164
    Base.precompile(Tuple{typeof(>),BigFloat,Int64})   # time: 0.001313632
    Base.precompile(Tuple{typeof(max),Float64,Int64})   # time: 0.00102575
end
