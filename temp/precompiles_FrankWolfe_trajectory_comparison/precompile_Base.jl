function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(getindex),Dict{Int32, Symbol},Int32})   # time: 0.015326115
    Base.precompile(Tuple{typeof(string),Int64})   # time: 0.012957649
    Base.precompile(Tuple{typeof(mod),Int64,Float64})   # time: 0.003943192
    Base.precompile(Tuple{Core.kwftype(typeof(with_output_color)),NamedTuple{(:bold, :underline, :blink, :reverse, :hidden), NTuple{5, Bool}},typeof(with_output_color),Function,Symbol,IOContext{IOBuffer},String})   # time: 0.003518972
    Base.precompile(Tuple{typeof(max),Float64,Int64})   # time: 0.001813858
    Base.precompile(Tuple{typeof(flush),TTY})   # time: 0.001320358
    Base.precompile(Tuple{typeof(/),UInt64,Float64})   # time: 0.001276288
end
