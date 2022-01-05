function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(-),Vector{Float64},Vector{Float64}})   # time: 0.015220323
    Base.precompile(Tuple{typeof(getindex),Dict{Int32, Symbol},Int32})   # time: 0.007173682
    Base.precompile(Tuple{typeof(string),Int64})   # time: 0.004495855
    Base.precompile(Tuple{typeof(mod),Int64,Float64})   # time: 0.00145883
    Base.precompile(Tuple{typeof(_isdisjoint),Tuple{UInt64},Tuple{UInt64, UInt64}})   # time: 0.001255177
end
