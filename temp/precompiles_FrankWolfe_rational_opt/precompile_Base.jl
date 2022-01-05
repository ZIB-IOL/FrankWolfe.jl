function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(setindex!),Vector{Rational{BigInt}},BigFloat,Int64})   # time: 0.00632933
    Base.precompile(Tuple{typeof(getindex),Dict{Int32, Symbol},Int32})   # time: 0.005373776
    Base.precompile(Tuple{typeof(string),Int64})   # time: 0.004368201
    Base.precompile(Tuple{typeof(copyto!),Vector{Int64},UnitRange{Int64}})   # time: 0.001721091
    Base.precompile(Tuple{typeof(mod),Int64,Float64})   # time: 0.001487126
    Base.precompile(Tuple{typeof(_isdisjoint),Tuple{UInt64, UInt64},Tuple{UInt64}})   # time: 0.001339759
end
