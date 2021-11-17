function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(dot),ScaledHotVector{Rational{BigInt}},ScaledHotVector{Rational{BigInt}}})   # time: 0.048759744
    isdefined(FrankWolfe, Symbol("#push_trajectory!#9")) && Base.precompile(Tuple{getfield(FrankWolfe, Symbol("#push_trajectory!#9")),NamedTuple{(:t, :primal, :dual, :dual_gap, :time, :x, :v, :gamma), _A} where _A<:Tuple{Int64, Union{Float64, Rational{BigInt}}, Union{Float64, Rational{BigInt}, BigFloat}, Union{Float64, Rational{BigInt}}, Float64, Union{ScaledHotVector{Rational{BigInt}}, SparseVector{Rational{BigInt}, Int64}}, ScaledHotVector{Rational{BigInt}}, Rational{Int64}}})   # time: 0.002432533
    Base.precompile(Tuple{Core.kwftype(typeof(print_callback)),NamedTuple{(:print_header,), Tuple{Bool}},typeof(print_callback),Vector{String},String})   # time: 0.00110002
end
