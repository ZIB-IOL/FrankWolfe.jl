function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(zip),SparseIntSet,SparseIntSet})   # time: 0.013791208
    Base.precompile(Tuple{typeof(iterate),ZippedSparseIntSetIterator{Tuple{SparseIntSet, SparseIntSet}, Tuple{}}})   # time: 0.007622986
end
