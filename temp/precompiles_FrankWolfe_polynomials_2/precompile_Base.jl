function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(getindex),Tuple{Int64, Float64, Float64, Float64, Float64, Vector{Float64}, Int64, Int64},UnitRange{Int64}})   # time: 0.12066655
    Base.precompile(Tuple{typeof(string),String,BigFloat})   # time: 0.07647221
    Base.precompile(Tuple{typeof(getindex),Dict{Int32, Symbol},Int32})   # time: 0.014141276
    Base.precompile(Tuple{typeof(maximum),Vector{Float64}})   # time: 0.011541267
    Base.precompile(Tuple{typeof(minimum),Vector{Float64}})   # time: 0.00966608
    Base.precompile(Tuple{typeof(string),String,Float64,String,Int64,String})   # time: 0.004193948
    Base.precompile(Tuple{typeof(max),Int64,BigFloat})   # time: 0.004108532
    Base.precompile(Tuple{typeof(!=),BigFloat,BigFloat})   # time: 0.002440771
    Base.precompile(Tuple{typeof(min),Int64,Float64})   # time: 0.002391043
    Base.precompile(Tuple{typeof(>),BigFloat,Int64})   # time: 0.002356614
    Base.precompile(Tuple{typeof(getindex),Tuple,UnitRange{Int64}})   # time: 0.002206029
    Base.precompile(Tuple{typeof(max),Int64,Float64})   # time: 0.002032471
    Base.precompile(Tuple{typeof(>),BigFloat,Float64})   # time: 0.001732644
    Base.precompile(Tuple{typeof(/),Float64,Int64})   # time: 0.001598056
    Base.precompile(Tuple{typeof(_similar_shape),UnitRange{Int64},HasShape{1}})   # time: 0.00142862
    Base.precompile(Tuple{typeof(max),Float64,Int64})   # time: 0.001387065
    Base.precompile(Tuple{typeof(/),UInt64,Float64})   # time: 0.001160498
    Base.precompile(Tuple{typeof(iterate),Tuple{String, DataType, String, Float64}})   # time: 0.001109932
    Base.precompile(Tuple{typeof(iterate),UnitRange{Int64}})   # time: 0.001018508
end
