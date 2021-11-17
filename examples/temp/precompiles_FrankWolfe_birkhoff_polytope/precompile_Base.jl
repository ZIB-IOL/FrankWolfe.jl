function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(string),String,BigFloat})   # time: 0.0307793
    Base.precompile(Tuple{typeof(iterate),UnitRange{Int64}})   # time: 0.021592453
    Base.precompile(Tuple{typeof(getindex),Dict{Int32, Symbol},Int32})   # time: 0.006796798
    Base.precompile(Tuple{typeof(maximum),Vector{Float64}})   # time: 0.005916542
    Base.precompile(Tuple{typeof(minimum),Vector{Float64}})   # time: 0.004629516
    Base.precompile(Tuple{typeof(print),IOBuffer,DataType})   # time: 0.0032835
    Base.precompile(Tuple{typeof(max),Int64,BigFloat})   # time: 0.002341087
    Base.precompile(Tuple{typeof(string),String,Float64,String,Int64,String})   # time: 0.002004163
    Base.precompile(Tuple{typeof(axes),UnitRange{Int64}})   # time: 0.001308174
    Base.precompile(Tuple{typeof(fill!),Vector{Float64},Any})   # time: 0.001044711
end
