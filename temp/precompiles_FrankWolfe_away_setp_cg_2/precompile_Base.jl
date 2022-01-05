function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(getindex),Dict{Int32, Symbol},Int32})   # time: 0.005932511
    Base.precompile(Tuple{typeof(string),Int64})   # time: 0.004885726
    Base.precompile(Tuple{typeof(flush),TTY})   # time: 0.001708312
    Base.precompile(Tuple{typeof(mod),Int64,Float64})   # time: 0.001668852
    Base.precompile(Tuple{Core.kwftype(typeof(with_output_color)),NamedTuple{(:bold, :underline, :blink, :reverse, :hidden), NTuple{5, Bool}},typeof(with_output_color),Function,Symbol,IOContext{IOBuffer},String})   # time: 0.001030993
end
