function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(getindex),Dict{Int32, Symbol},Int32})   # time: 0.005388437
    Base.precompile(Tuple{typeof(string),Int64})   # time: 0.004113497
    Base.precompile(Tuple{typeof(promote_shape),Tuple{OneTo{Int64}},Tuple{OneTo{Int64}}})   # time: 0.002351125
    Base.precompile(Tuple{typeof(mod),Int64,Float64})   # time: 0.001272434
    Base.precompile(Tuple{Core.kwftype(typeof(with_output_color)),NamedTuple{(:bold, :underline, :blink, :reverse, :hidden), NTuple{5, Bool}},typeof(with_output_color),Function,Symbol,IOContext{IOBuffer},String})   # time: 0.001121646
    Base.precompile(Tuple{typeof(-),BigFloat,Rational{BigInt}})   # time: 0.001110053
end
