function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(default_metafmt),Base.CoreLogging.LogLevel,Any,Any,Any,Any,Any})   # time: 0.002229673
end
