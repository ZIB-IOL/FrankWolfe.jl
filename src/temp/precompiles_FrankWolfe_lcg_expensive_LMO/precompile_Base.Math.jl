function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(min),Float64,Float64})   # time: 0.001135127
    Base.precompile(Tuple{typeof(max),Float64,Float64})   # time: 0.001078834
end
