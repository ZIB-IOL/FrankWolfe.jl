function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(_zeros_eltypes),AbstractSparseMatrixCSC})   # time: 0.030701887
    Base.precompile(Tuple{typeof(_fusedupdate_all),Int64,Any,Tuple{Any, Vararg{Any, N} where N},Tuple{Integer, Vararg{Any, N} where N},Tuple{Integer, Vararg{Any, N} where N},Tuple{AbstractSparseMatrixCSC, Vararg{Union{AbstractSparseMatrixCSC, SparseVector}, N} where N}})   # time: 0.00811226
    Base.precompile(Tuple{typeof(_unchecked_maxnnzbcres),Tuple{Int64, Int64},Tuple{Vararg{Union{AbstractSparseMatrixCSC, SparseVector}, N} where N}})   # time: 0.006312731
    Base.precompile(Tuple{typeof(_allocres),Tuple{Int64, Int64},Type{Int64},Type{BigFloat},Int64})   # time: 0.004747283
    isdefined(SparseArrays.HigherOrderFns, Symbol("#3#4")) && Base.precompile(Tuple{getfield(SparseArrays.HigherOrderFns, Symbol("#3#4")),Float64,Float64})   # time: 0.004708813
    Base.precompile(Tuple{typeof(_fusedupdatebc_all),Int64,Any,Tuple,Tuple,Tuple,Tuple,Tuple{Vararg{Union{AbstractSparseMatrixCSC, SparseVector}, N} where N}})   # time: 0.003931992
    Base.precompile(Tuple{typeof(_rowforind_all),Int64,Tuple{Integer, Vararg{Any, N} where N},Tuple{Integer, Vararg{Any, N} where N},Tuple{AbstractSparseMatrixCSC, Vararg{Union{AbstractSparseMatrixCSC, SparseVector}, N} where N}})   # time: 0.003347372
    Base.precompile(Tuple{typeof(_sparsifystructured),Matrix{Float64}})   # time: 0.003062707
    Base.precompile(Tuple{typeof(_fusedupdate_all),Int64,Any,Tuple{Any, Vararg{Any, N} where N},Tuple{Any, Vararg{Any, N} where N},Tuple{Integer, Vararg{Any, N} where N},Tuple{AbstractSparseMatrixCSC, Vararg{Union{AbstractSparseMatrixCSC, SparseVector}, N} where N}})   # time: 0.002037058
    Base.precompile(Tuple{typeof(_colstartind_all),Int64,Tuple{AbstractSparseMatrixCSC, Vararg{Union{AbstractSparseMatrixCSC, SparseVector}, N} where N}})   # time: 0.001848839
    Base.precompile(Tuple{typeof(_stopindforbccol_all),Int64,Tuple{Vararg{Bool, N} where N},Tuple{Vararg{Union{AbstractSparseMatrixCSC, SparseVector}, N} where N}})   # time: 0.001585689
    Base.precompile(Tuple{typeof(_aresameshape),AbstractSparseMatrixCSC,Union{AbstractSparseMatrixCSC, SparseVector},Vararg{Union{AbstractSparseMatrixCSC, SparseVector}, N} where N})   # time: 0.001584989
    Base.precompile(Tuple{typeof(_capturescalars),Base.RefValue{typeof(^)},Int64,Base.RefValue{Val{2}}})   # time: 0.001445801
    isdefined(SparseArrays.HigherOrderFns, Symbol("#3#4")) && Base.precompile(Tuple{getfield(SparseArrays.HigherOrderFns, Symbol("#3#4")),Float64,Float64})   # time: 0.001314016
    Base.precompile(Tuple{typeof(_unchecked_maxnnzbcres),Tuple{Int64, Int64},AbstractSparseMatrixCSC})   # time: 0.00117308
    Base.precompile(Tuple{typeof(_initrowforcol_all),Int64,Int64,Tuple,Tuple{Vararg{Bool, N} where N},Tuple,Tuple{Vararg{Union{AbstractSparseMatrixCSC, SparseVector}, N} where N}})   # time: 0.001143958
    Base.precompile(Tuple{typeof(_colstartind_all),Int64,Tuple{SparseVector, Vararg{Union{AbstractSparseMatrixCSC, SparseVector}, N} where N}})   # time: 0.001112671
    Base.precompile(Tuple{typeof(_rowforind_all),Int64,Tuple{Any, Vararg{Any, N} where N},Tuple{Any, Vararg{Any, N} where N},Tuple{SparseVector, Vararg{Union{AbstractSparseMatrixCSC, SparseVector}, N} where N}})   # time: 0.001108412
    Base.precompile(Tuple{typeof(_unchecked_maxnnzbcres),Tuple{Int64, Int64},SparseVector})   # time: 0.001030888
    Base.precompile(Tuple{typeof(_unchecked_maxnnzbcres),Tuple{Int64, Int64},AbstractSparseMatrixCSC,AbstractSparseMatrixCSC})   # time: 0.001009243
end
