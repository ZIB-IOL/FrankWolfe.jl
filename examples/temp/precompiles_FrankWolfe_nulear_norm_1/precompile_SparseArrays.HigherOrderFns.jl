function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(_allocres),Tuple{Int64, Int64},Type{Int64},Type{BigFloat},Int64})   # time: 0.007806737
    Base.precompile(Tuple{typeof(_iszero),BigFloat})   # time: 0.001617778
end
