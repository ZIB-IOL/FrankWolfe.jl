function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(isapprox),BigFloat,Float64})   # time: 0.0069466
    Base.precompile(Tuple{typeof(getindex),Dict{Int32, Symbol},Int32})   # time: 0.006090776
    Base.precompile(Tuple{typeof(string),Int64})   # time: 0.00449715
    Base.precompile(Tuple{typeof(string),String,Type{Nothing},String,Bool,String,Float64})   # time: 0.004138296
    Base.precompile(Tuple{typeof(isapprox),Float64,Float64})   # time: 0.001823688
    Base.precompile(Tuple{typeof(isapprox),Int64,Float64})   # time: 0.001443196
    Base.precompile(Tuple{typeof(max),Int64,Float64})   # time: 0.001286469
    Base.precompile(Tuple{typeof(axes),UnitRange{Int64}})   # time: 0.001159223
end
