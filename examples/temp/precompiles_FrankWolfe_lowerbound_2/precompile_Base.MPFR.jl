function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(-),BigFloat,Float64})   # time: 0.001081196
    Base.precompile(Tuple{typeof(*),Float64,BigFloat})   # time: 0.001025604
end
