function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(_promote_tuple_shape),Tuple{OneTo{Int64}},Tuple{OneTo{Int64}}})   # time: 0.001549278
end
