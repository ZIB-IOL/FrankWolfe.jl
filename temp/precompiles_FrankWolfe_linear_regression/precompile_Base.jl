function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(getindex),Dict{Int32, Symbol},Int32})   # time: 0.015831867
    Base.precompile(Tuple{typeof(string),Int64})   # time: 0.011111067
    Base.precompile(Tuple{typeof(mod),Int64,Float64})   # time: 0.003582332
    Base.precompile(Tuple{Core.kwftype(typeof(with_output_color)),NamedTuple{(:bold, :underline, :blink, :reverse, :hidden), NTuple{5, Bool}},typeof(with_output_color),Function,Symbol,IOContext{IOBuffer},String})   # time: 0.00224076
    Base.precompile(Tuple{typeof(iterate),Tuple{String, DataType, String, Bool, String, DataType, String}})   # time: 0.001478805
    Base.precompile(Tuple{typeof(flush),TTY})   # time: 0.00147866
    Base.precompile(Tuple{typeof(/),UInt64,Float64})   # time: 0.001290232
end
