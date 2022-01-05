function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(getindex),Dict{Int32, Symbol},Int32})   # time: 0.008090489
    Base.precompile(Tuple{typeof(string),Int64})   # time: 0.006976602
    Base.precompile(Tuple{typeof(max),Int64,Float64})   # time: 0.00162366
    Base.precompile(Tuple{typeof(max),Float64,Int64})   # time: 0.001236601
    Base.precompile(Tuple{typeof(mod),Int64,Float64})   # time: 0.001037764
end
