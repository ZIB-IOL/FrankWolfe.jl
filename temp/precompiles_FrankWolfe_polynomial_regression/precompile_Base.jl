function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(string),Int64})   # time: 0.030200023
    Base.precompile(Tuple{typeof(getindex),Dict{Int32, Symbol},Int32})   # time: 0.006382901
    Base.precompile(Tuple{typeof(string),String,Type{Vector{Float64}},String,Bool,String,Float64,String,Nothing,String,Bool})   # time: 0.003833911
    Base.precompile(Tuple{typeof(getindex),Tuple,UnitRange{Int64}})   # time: 0.00112157
    Base.precompile(Tuple{typeof(max),Int64,Float64})   # time: 0.00101765
end
