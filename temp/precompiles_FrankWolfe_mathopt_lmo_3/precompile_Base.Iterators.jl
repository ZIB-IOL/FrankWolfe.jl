function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(iterate),Zip,Int64})   # time: 0.003182439
    Base.precompile(Tuple{typeof(zip),Any,Any})   # time: 0.001868244
end
