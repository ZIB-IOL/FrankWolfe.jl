function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(show),IOBuffer,Type})   # time: 0.079854734
    Base.precompile(Tuple{typeof(getindex),Dict{Int32, Symbol},Int32})   # time: 0.04364996
    Base.precompile(Tuple{typeof(string),Int64})   # time: 0.004958445
    Base.precompile(Tuple{typeof(axes),UnitRange{Int64}})   # time: 0.001187719
    Base.precompile(Tuple{typeof(max),Int64,Float64})   # time: 0.001067453
end
