function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(getindex),Dict{Int32, Symbol},Int32})   # time: 0.015144372
    Base.precompile(Tuple{typeof(string),Int64})   # time: 0.012592674
    Base.precompile(Tuple{Type{IteratorEltype},Vector{Tuple{Vector{Float64}, Float64}}})   # time: 0.006067048
    Base.precompile(Tuple{typeof(mod),Int64,Float64})   # time: 0.004108299
    Base.precompile(Tuple{typeof(max),Float64,Int64})   # time: 0.002077125
    Base.precompile(Tuple{typeof(/),Float64,Int64})   # time: 0.001424329
    Base.precompile(Tuple{typeof(/),UInt64,Float64})   # time: 0.001368596
    Base.precompile(Tuple{Colon,Int64,Int64})   # time: 0.001180305
end
