function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(show),IOBuffer,Type})   # time: 0.13120006
    Base.precompile(Tuple{typeof(getindex),Dict{Int32, Symbol},Int32})   # time: 0.013411937
    Base.precompile(Tuple{typeof(string),Int64})   # time: 0.011677218
    Base.precompile(Tuple{typeof(max),Float64,Int64})   # time: 0.002488928
    Base.precompile(Tuple{typeof(mod),Int64,Float64})   # time: 0.00202148
    Base.precompile(Tuple{typeof(/),UInt64,Float64})   # time: 0.00141582
end
