"""
    primal_value_stop(f_limit)

Checks if the primal objective is surpasses the value f_limit.
If so, sets the stop_criterion to true 
"""
function primal_value_stop(primal_limit)
    return function limit_check(state) 
        return state.primal < primal_limit
    end
end

"""
    f_call_stop(f_call_limit)

Checks if the gradient call count surpasses the value grad_limit.
If so, sets the stop_criterion to true 
"""
function f_call_stop(f_call_limit)
    return function limit_check(state) 
        return state.f.counter < f_call_limit
    end
end

"""
    grad_call_stop(grad_call_limit)

Checks if the gradient call count surpasses the value grad_limit.
If so, sets the stop_criterion to true 
"""
function grad_call_stop(grad_call_limit)
    return function limit_check(state) 
        return state.grad.counter < grad_call_limit
    end
end

"""
    lmo_call_stop(lmo_call_limit)

Checks if the lmo call count surpasses the value lmo_limit.
If so, sets the stop_criterion to true 
"""
function lmo_call_stop(lmo_call_limit)
    return function limit_check(state) 
        return state.lmo.counter < lmo_call_limit
    end
end

function push_state(state,storage)
    base_tuple = Tuple(state)[1:5]
    if typeof(state.lmo) <: CachedLinearMinimizationOracle
        complete_tuple = tuple(base_tuple..., state.gamma, state.f.counter, state.grad.counter, state.lmo.inner.counter)
    else
        complete_tuple = tuple(base_tuple..., state.gamma, state.f.counter, state.grad.counter, state.lmo.counter)
    end
    push!(storage, complete_tuple)
end

function print_callback(data, format_string; print_header=false, print_footer=false)
    print_formatted(fmt, args...) = @eval @printf($fmt, $(args...))
    if print_header || print_footer
        temp = strip(format_string, ['\n'])
        temp = replace(temp, "%" => "")
        temp = replace(temp, "e" => "")
        temp = replace(temp, "i" => "")
        temp = replace(temp, "s" => "")
        temp = split(temp, " ")
        len = 0
        for i in temp
            len = len + parse(Int, i)
        end
        lenHeaderFooter = len + 2 + length(temp) - 1
        if print_footer
            line = "-"^lenHeaderFooter
            @printf("%s\n\n", line)
        end
        if print_header
            line = "-"^lenHeaderFooter
            @printf("\n%s\n", line)
            s_format_string = replace(format_string, "e" => "s")
            s_format_string = replace(s_format_string, "i" => "s")
            print_formatted(s_format_string, data...)
            @printf("%s\n", line)
        end
    else
        print_formatted(format_string, data...)
    end
end

"""
    Callback for fw_algorithms
    If verbose is true, prints the state to the console after print_iter iterations.
    If trajectory is true, adds the state to the storage variable.
    The state data is only the 5 first fields, gamma and 3 call counters, usually
`(t, primal, dual, dual_gap, time, gamma, function_calls, gradient_calls, lmo_calls)`
"""
function make_callback(traj_data, stop_criterion, verbose, trajectory, print_iter, headers, format_string, format_state)
    return function callback(state)
        if trajectory && (state.tt !== last)
            push_state(state, traj_data)
        end
        if verbose 
            if mod(state.t, print_iter) == 0
                if state.t == 0
                    state = merge(state,(tt=initial,))
                    print_callback(headers, format_string, print_header=true)
                end
                rep = format_state(state)
                print_callback(rep, format_string)
                flush(stdout)
            end 

            if (state.tt == last)
                rep = format_state(state)
                print_callback(rep, format_string)
                print_callback(nothing, format_string, print_footer=true)
                flush(stdout)
            end
        end
        
        if stop_criterion !== nothing
            return stop_criterion(state)
        else 
            return false
        end
    end
end