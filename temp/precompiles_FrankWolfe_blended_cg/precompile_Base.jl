function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(_minimum),Vector{Float64},Colon})   # time: 0.009413457
    Base.precompile(Tuple{typeof(-),Int64,Float64})   # time: 0.00156018
    Base.precompile(Tuple{typeof(flush),TTY})   # time: 0.001148046
end
