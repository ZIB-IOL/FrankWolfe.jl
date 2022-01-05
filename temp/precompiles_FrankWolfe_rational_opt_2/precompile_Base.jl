function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(getindex),Dict{Int32, Symbol},Int32})   # time: 0.005525453
    Base.precompile(Tuple{typeof(string),Int64})   # time: 0.00450341
    Base.precompile(Tuple{typeof(//),Int64,Int64})   # time: 0.003960887
    Base.precompile(Tuple{typeof(mod),Int64,Float64})   # time: 0.001464005
    Base.precompile(Tuple{typeof(max),Rational{BigInt},Int64})   # time: 0.001208039
end
