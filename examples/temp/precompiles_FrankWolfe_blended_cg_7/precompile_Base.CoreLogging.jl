function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(current_logger_for_env),LogLevel,Symbol,Module})   # time: 0.006624001
end
