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

"""
A function acting like the passed callback,
    but adding the state to the storage variable.
    The state data is only the 5 first fields, gamma and 3 call counters, usually
`(t, primal, dual, dual_gap, time, gamma, function_calls, gradient_calls, lmo_calls)`
"""
function tracking_callback(storage, callback)
    function push_and_callback(state)
        base_tuple = Tuple(state)[1:5]
        complete_tuple = tuple(base_tuple..., state.gamma, state.f.counter, state.grad.counter, state.lmo.counter)
        push!(storage, complete_tuple)
        return callback(state)
    end
end

tracking_callback(storage) = tracking_callback(storage, state->false)

"""
A function acting like the passed callback for cached LMOs,
    but adding the state to the storage variable.
    The state data is only the 5 first fields, gamma and 3 call counters, usually
`(t, primal, dual, dual_gap, time, gamma, function_calls, gradient_calls, lmo_calls)`
"""
function tracking_cached_callback(storage, stop_criterion)
    function push_and_callback(state)
        base_tuple = Tuple(state)[1:5]
        complete_tuple = tuple(base_tuple..., state.gamma, state.f.counter, state.grad.counter, state.lmo.inner.counter)
        push!(storage, complete_tuple)
        return stop_criterion(state)
    end
end

tracking_cached_callback(storage) = tracking_cached_callback(storage, state->false)
