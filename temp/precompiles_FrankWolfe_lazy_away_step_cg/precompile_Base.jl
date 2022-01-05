function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(getindex),Dict{Int32, Symbol},Int32})   # time: 0.05577639
    Base.precompile(Tuple{typeof(string),Int64})   # time: 0.009752304
    Base.precompile(Tuple{typeof(copyto_unaliased!),IndexLinear,SubArray{Int64, 1, Vector{Int64}, Tuple{UnitRange{Int64}}, true},IndexLinear,Vector{Int64}})   # time: 0.005012592
    Base.precompile(Tuple{Core.kwftype(typeof(with_output_color)),NamedTuple{(:bold, :underline, :blink, :reverse, :hidden), NTuple{5, Bool}},typeof(with_output_color),Function,Symbol,IOContext{IOBuffer},String})   # time: 0.001973133
    Base.precompile(Tuple{typeof(max),Int64,Float64})   # time: 0.001627843
    Base.precompile(Tuple{typeof(mod),Int64,Float64})   # time: 0.001561283
    Base.precompile(Tuple{typeof(flush),TTY})   # time: 0.00120803
    Base.precompile(Tuple{typeof(mod),Int64,Int64})   # time: 0.001045024
end
