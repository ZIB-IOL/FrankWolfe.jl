function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(iterate),Zip,Int64})   # time: 0.004741593
    Base.precompile(Tuple{typeof(zip),Any,Any})   # time: 0.003102092
    Base.precompile(Tuple{typeof(zip),Vector{DataType},Vector{_A} where _A})   # time: 0.002130613
end
