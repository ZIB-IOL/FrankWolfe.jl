function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(getindex),Dict{Int32, Symbol},Int32})   # time: 0.049242858
    Base.precompile(Tuple{typeof(string),Int64})   # time: 0.004091978
    Base.precompile(Tuple{typeof(push!),Vector{Tuple{Int64, Float64, Float64, Float64, Float64, Float64}},Tuple{Any, Any, Any, Any, Any, Float64}})   # time: 0.003636969
    Base.precompile(Tuple{typeof(vect),String,String,String,String,String,String,String})   # time: 0.003422764
    Base.precompile(Tuple{typeof(mod),Int64,Float64})   # time: 0.001345203
    Base.precompile(Tuple{typeof(getindex),Tuple,UnitRange{Int64}})   # time: 0.001279975
end
