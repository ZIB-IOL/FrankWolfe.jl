function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(getindex),Dict{Int32, Symbol},Int32})   # time: 0.008642581
    Base.precompile(Tuple{typeof(string),Int64})   # time: 0.007631839
    Base.precompile(Tuple{typeof(mod),Int64,Float64})   # time: 0.003003849
    Base.precompile(Tuple{Core.kwftype(typeof(with_output_color)),NamedTuple{(:bold, :underline, :blink, :reverse, :hidden), NTuple{5, Bool}},typeof(with_output_color),Function,Symbol,IOContext{IOBuffer},String})   # time: 0.001671706
    Base.precompile(Tuple{typeof(max),Float64,Int64})   # time: 0.001269282
    Base.precompile(Tuple{typeof(flush),TTY})   # time: 0.001232827
end
