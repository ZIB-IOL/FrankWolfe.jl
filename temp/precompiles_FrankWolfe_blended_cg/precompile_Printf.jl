function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(fmt),Vector{UInt8},Int64,Int64,Spec{Val{'i'}}})   # time: 0.011885276
end
