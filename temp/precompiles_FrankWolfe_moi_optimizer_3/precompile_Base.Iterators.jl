function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(iterate),Zip,Int64})   # time: 0.005412168
    Base.precompile(Tuple{typeof(zip),Any,Any})   # time: 0.002570835
end
