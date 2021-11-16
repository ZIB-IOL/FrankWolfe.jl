function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(getindex),Dict{Int32, Symbol},Int32})   # time: 0.005324615
    Base.precompile(Tuple{typeof(string),Int64})   # time: 0.00435131
    Base.precompile(Tuple{typeof(axes),UnitRange{Int64}})   # time: 0.001245408
    Base.precompile(Tuple{typeof(max),Int64,Float64})   # time: 0.001057742
end
