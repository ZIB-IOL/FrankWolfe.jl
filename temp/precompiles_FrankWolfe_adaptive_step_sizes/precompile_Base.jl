function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(getindex),Dict{Int32, Symbol},Int32})   # time: 0.006409311
    Base.precompile(Tuple{typeof(string),Int64})   # time: 0.005172289
    Base.precompile(Tuple{typeof(mod),Int64,Float64})   # time: 0.001927329
    Base.precompile(Tuple{typeof(flush),TTY})   # time: 0.001246035
    Base.precompile(Tuple{typeof(max),Float64,Int64})   # time: 0.001156847
end
