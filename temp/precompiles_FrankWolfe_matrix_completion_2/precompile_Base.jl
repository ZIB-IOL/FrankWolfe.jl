function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(getindex),Dict{Int32, Symbol},Int32})   # time: 0.005865237
    Base.precompile(Tuple{typeof(string),Int64})   # time: 0.004820133
    Base.precompile(Tuple{typeof(vect),String,String,String,String,String,String,String,String})   # time: 0.003370101
    Base.precompile(Tuple{typeof(mod),Int64,Float64})   # time: 0.001354353
end
