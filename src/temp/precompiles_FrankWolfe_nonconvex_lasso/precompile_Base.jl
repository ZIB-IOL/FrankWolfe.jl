function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(getindex),Dict{Int32, Symbol},Int32})   # time: 0.014014538
    Base.precompile(Tuple{typeof(string),Int64})   # time: 0.011357693
    Base.precompile(Tuple{typeof(mod),Int64,Float64})   # time: 0.002382002
    Base.precompile(Tuple{typeof(/),UInt64,Float64})   # time: 0.001716486
    Base.precompile(Tuple{typeof(/),Int64,Float64})   # time: 0.001327402
end
