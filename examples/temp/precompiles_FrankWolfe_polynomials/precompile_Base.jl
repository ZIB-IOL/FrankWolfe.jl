function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(getindex),Dict{Int32, Symbol},Int32})   # time: 0.018161166
    Base.precompile(Tuple{typeof(string),Int64})   # time: 0.013447269
    Base.precompile(Tuple{typeof(string),String,Type{Vector{Float64}},String,Bool,String,Float64,String,Nothing,String,Bool})   # time: 0.008332903
    Base.precompile(Tuple{typeof(_compute_eltype),Any})   # time: 0.005441291
    Base.precompile(Tuple{typeof(getindex),Tuple,UnitRange{Int64}})   # time: 0.004851693
    Base.precompile(Tuple{typeof(_isdisjoint),Tuple{UInt64, UInt64},Tuple{UInt64, UInt64}})   # time: 0.003867499
    Base.precompile(Tuple{typeof(_array_for),Type{Any},UnitRange{Int64},HasShape{1}})   # time: 0.003079
    Base.precompile(Tuple{typeof(axes),UnitRange{Int64}})   # time: 0.002963831
    Base.precompile(Tuple{typeof(>),BigFloat,Float64})   # time: 0.002403987
    Base.precompile(Tuple{typeof(mapreduce_impl),typeof(identity),typeof(add_sum),Vector{Float64},Int64,Int64,Int64})   # time: 0.002369761
    Base.precompile(Tuple{typeof(mod),Int64,Int64})   # time: 0.002160244
    Base.precompile(Tuple{typeof(max),Int64,Float64})   # time: 0.00213908
    Base.precompile(Tuple{typeof(/),Float64,Int64})   # time: 0.001668354
    Base.precompile(Tuple{typeof(min),Float64,Int64})   # time: 0.00151184
    Base.precompile(Tuple{typeof(/),UInt64,Float64})   # time: 0.00147818
    Base.precompile(Tuple{typeof(>),Float64,BigFloat})   # time: 0.00129764
    Base.precompile(Tuple{typeof(iterate),IdSet{Any},Int64})   # time: 0.001165574
    Base.precompile(Tuple{typeof(flush),TTY})   # time: 0.001077991
end
