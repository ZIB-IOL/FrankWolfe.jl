function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(_invoked_shouldlog),AbstractLogger,LogLevel,Module,Symbol,Symbol})   # time: 0.08457674
    Base.precompile(Tuple{typeof(current_logger_for_env),LogLevel,Symbol,Module})   # time: 0.018537352
end
