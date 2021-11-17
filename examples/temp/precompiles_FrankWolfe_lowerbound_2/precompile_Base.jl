function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(min),Float64,Int64})   # time: 0.054664135
    Base.precompile(Tuple{typeof(*),Float64,Vector{BigFloat}})   # time: 0.040319376
    Base.precompile(Tuple{typeof(*),BigFloat,Vector{BigFloat}})   # time: 0.033823736
    Base.precompile(Tuple{typeof(*),BigFloat,Vector{Float64}})   # time: 0.029673977
    Base.precompile(Tuple{typeof(-),Vector{Float64}})   # time: 0.022289157
    Base.precompile(Tuple{typeof(getindex),Dict{Int32, Symbol},Int32})   # time: 0.014757159
    Base.precompile(Tuple{typeof(string),Int64})   # time: 0.010719278
    Base.precompile(Tuple{typeof(mapreduce_impl),typeof(identity),typeof(add_sum),Vector{Float64},Int64,Int64,Int64})   # time: 0.003122757
    Base.precompile(Tuple{typeof(max),BigFloat,Float64})   # time: 0.00259114
    Base.precompile(Tuple{typeof(min),BigFloat,Int64})   # time: 0.002399845
    Base.precompile(Tuple{typeof(max),Int64,Float64})   # time: 0.001458333
    Base.precompile(Tuple{typeof(mod),Int64,Int64})   # time: 0.001450094
    Base.precompile(Tuple{typeof(min),BigFloat,Float64})   # time: 0.001257962
    Base.precompile(Tuple{typeof(mod),Int64,Float64})   # time: 0.001117165
end
