function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(fpsort!),Vector{Float64},Base.Sort.QuickSortAlg,ReverseOrdering{ForwardOrdering}})   # time: 0.006598459
end
