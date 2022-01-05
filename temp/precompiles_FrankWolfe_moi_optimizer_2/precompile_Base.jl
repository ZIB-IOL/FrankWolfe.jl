function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(mod),Int64,Float64})   # time: 0.0454385
    Base.precompile(Tuple{typeof(getindex),Dict{Int32, Symbol},Int32})   # time: 0.012573138
    Base.precompile(Tuple{typeof(string),Int64})   # time: 0.009927156
    Base.precompile(Tuple{typeof(/),UInt64,Float64})   # time: 0.002300062
    Base.precompile(Tuple{typeof(max),Float64,Int64})   # time: 0.001711025
    Base.precompile(Tuple{typeof(similar),Type{Array{Float64}},Tuple{OneTo{Int64}}})   # time: 0.001460999
    Base.precompile(Tuple{typeof(_array_for),Type{HasShape{1}},HasShape{1},Tuple{OneTo{Int64}}})   # time: 0.001099502
end
