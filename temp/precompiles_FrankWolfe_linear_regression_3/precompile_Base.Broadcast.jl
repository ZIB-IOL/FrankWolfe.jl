function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(materialize!),SubArray{Float64, 1, Vector{Float64}, Tuple{UnitRange{Int64}}, true},Broadcasted{Style}})   # time: 0.046917837
    Base.precompile(Tuple{typeof(broadcasted),typeof(+),Vector{Float64},Any})   # time: 0.034472108
    Base.precompile(Tuple{typeof(dotview),Vector{Float64},UnitRange{Int64}})   # time: 0.006531625
    Base.precompile(Tuple{typeof(materialize!),Vector{Float64},Broadcasted{DefaultArrayStyle{0}, Nothing, typeof(identity), Tuple{Int64}}})   # time: 0.005363965
    Base.precompile(Tuple{typeof(broadcasted),typeof(identity),Int64})   # time: 0.003435459
    Base.precompile(Tuple{typeof(broadcasted),typeof(-),Int64,Int64})   # time: 0.003229847
    Base.precompile(Tuple{typeof(materialize),Broadcasted{DefaultArrayStyle{0}, Nothing, typeof(-), Tuple{Int64, Int64}}})   # time: 0.003151342
end
