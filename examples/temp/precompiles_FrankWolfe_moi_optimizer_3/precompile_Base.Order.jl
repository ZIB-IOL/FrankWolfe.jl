function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(lt),ForwardOrdering,Tuple{Float64, Bool},Tuple{Float64, Bool}})   # time: 0.001228278
end
