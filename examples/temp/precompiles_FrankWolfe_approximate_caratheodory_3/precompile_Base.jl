function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(string),Int64})   # time: 0.029129313
    Base.precompile(Tuple{typeof(getindex),Dict{Int32, Symbol},Int32})   # time: 0.005951472
    Base.precompile(Tuple{typeof(promote_shape),Tuple{OneTo{Int64}},Tuple{OneTo{Int64}}})   # time: 0.00198857
    Base.precompile(Tuple{typeof(-),BigFloat,Rational{BigInt}})   # time: 0.0013165
end
