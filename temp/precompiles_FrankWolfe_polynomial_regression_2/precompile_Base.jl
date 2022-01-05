function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(getindex),Tuple{Int64, Float64, Float64, Float64, Float64, Vector{Float64}, Int64, Int64},UnitRange{Int64}})   # time: 0.09495313
    Base.precompile(Tuple{typeof(string),String,BigFloat})   # time: 0.033397064
    Base.precompile(Tuple{typeof(getindex),Dict{Int32, Symbol},Int32})   # time: 0.006224781
    Base.precompile(Tuple{typeof(maximum),Vector{Float64}})   # time: 0.005326911
    Base.precompile(Tuple{typeof(minimum),Vector{Float64}})   # time: 0.004306801
    Base.precompile(Tuple{typeof(string),String,Float64,String,Int64,String})   # time: 0.002205786
    Base.precompile(Tuple{typeof(max),Int64,BigFloat})   # time: 0.00185728
    Base.precompile(Tuple{typeof(getindex),Tuple,UnitRange{Int64}})   # time: 0.00119477
end
