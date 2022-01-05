function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(default_metafmt),Base.CoreLogging.LogLevel,Any,Any,Any,Any,Any})   # time: 0.003850304
    Base.precompile(Tuple{typeof(Base.CoreLogging.shouldlog),ConsoleLogger,Base.CoreLogging.LogLevel,Module,Symbol,Symbol})   # time: 0.002186216
end
