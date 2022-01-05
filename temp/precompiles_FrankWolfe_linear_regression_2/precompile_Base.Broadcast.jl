function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(materialize!),Vector{Float64},Broadcasted{DefaultArrayStyle{1}, Nothing, typeof(+), Tuple{Broadcasted{DefaultArrayStyle{1}, Nothing, typeof(*), Tuple{Broadcasted{DefaultArrayStyle{1}, Nothing, typeof(*), Tuple{Bool, Vector{Float64}}}, Float64}}, Broadcasted{DefaultArrayStyle{1}, Nothing, typeof(*), Tuple{Vector{Float64}, Float64}}}}})   # time: 0.11049783
    Base.precompile(Tuple{typeof(materialize!),Vector{Float64},Broadcasted{DefaultArrayStyle{1}, Nothing, typeof(+), Tuple{Broadcasted{DefaultArrayStyle{1}, Nothing, typeof(*), Tuple{Broadcasted{DefaultArrayStyle{0}, Nothing, typeof(-), Tuple{Int64, Float64}}, Vector{Float64}}}, Broadcasted{DefaultArrayStyle{1}, Nothing, typeof(*), Tuple{Float64, Vector{Float64}}}}}})   # time: 0.0338758
    Base.precompile(Tuple{typeof(broadcasted),typeof(-),Int64,Float64})   # time: 0.003884671
    Base.precompile(Tuple{typeof(broadcasted),typeof(+),Broadcasted{DefaultArrayStyle{1}, Nothing, typeof(*), Tuple{Broadcasted{DefaultArrayStyle{0}, Nothing, typeof(-), Tuple{Int64, Float64}}, Vector{Float64}}},Broadcasted{DefaultArrayStyle{1}, Nothing, typeof(*), Tuple{Float64, Vector{Float64}}}})   # time: 0.003313302
    Base.precompile(Tuple{typeof(broadcasted),typeof(*),Broadcasted{DefaultArrayStyle{0}, Nothing, typeof(-), Tuple{Int64, Float64}},Vector{Float64}})   # time: 0.003262879
end
