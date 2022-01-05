function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(-),Vector{Float64},Vector{Float64}})   # time: 0.006230369
    Base.precompile(Tuple{typeof(getindex),Dict{Int32, Symbol},Int32})   # time: 0.005235418
    Base.precompile(Tuple{typeof(string),Int64})   # time: 0.004152661
    Base.precompile(Tuple{typeof(max),Float64,Int64})   # time: 0.001123878
    Base.precompile(Tuple{Core.kwftype(typeof(with_output_color)),NamedTuple{(:bold, :underline, :blink, :reverse, :hidden), NTuple{5, Bool}},typeof(with_output_color),Function,Symbol,IOContext{IOBuffer},String})   # time: 0.001002078
end
