function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(zip),SparseIntSet,SparseIntSet})   # time: 0.016996082
    Base.precompile(Tuple{typeof(iterate),ZippedSparseIntSetIterator{Tuple{SparseIntSet, SparseIntSet}, Tuple{}}})   # time: 0.005124367
end
