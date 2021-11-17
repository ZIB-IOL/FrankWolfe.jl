function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{Core.kwftype(typeof(print_callback)),NamedTuple{(:print_header,), Tuple{Bool}},typeof(print_callback),NTuple{8, String},String})   # time: 0.06601599
    Base.precompile(Tuple{typeof(print_callback),Tuple{String, String, Float64, Float64, Float64, Float64, Float64, Int64},String})   # time: 0.001409586
end
