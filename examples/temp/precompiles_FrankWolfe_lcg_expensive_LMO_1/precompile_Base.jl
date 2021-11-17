function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{Type{NamedTuple{(:gamma_max,), _A}} where _A<:Tuple{Union{Float64, BigFloat}},Tuple{Union{Float64, BigFloat}}})   # time: 0.03620036
    Base.precompile(Tuple{typeof(getindex),Dict{Int32, Symbol},Int32})   # time: 0.006055711
    Base.precompile(Tuple{typeof(string),Int64})   # time: 0.004635748
    Base.precompile(Tuple{typeof(string),String,Nothing,String,Type{Nothing}})   # time: 0.004287684
    Base.precompile(Tuple{typeof(structdiff),NamedTuple{(:gamma_max,), _A} where _A<:Tuple{Union{Float64, BigFloat}},Type{NamedTuple{(:eta, :tau, :gamma_max, :upgrade_accuracy), T} where T<:Tuple}})   # time: 0.004251713
    Base.precompile(Tuple{typeof(vect),String,String,String,String,String,String,String})   # time: 0.003961535
    Base.precompile(Tuple{typeof(lstrip),Fix2{typeof(in), Vector{Char}},SubString{String}})   # time: 0.002745082
    Base.precompile(Tuple{typeof(rstrip),Fix2{typeof(in), Vector{Char}},String})   # time: 0.002360549
    Base.precompile(Tuple{typeof(max),BigFloat,Float64})   # time: 0.001462355
end
