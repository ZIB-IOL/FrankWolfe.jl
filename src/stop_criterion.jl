"""
    primal_value_callback(f_limit)

Callback checking if the primal objective is surpasses the value f_limit.
If so, sets the stop_criterion to true 
"""
function primal_value_callback(primal_limit)
    return function limit_check!(state) 
        return state.primal < primal_limit
    end
end

"""
    f_call_callback(f_call_limit)

Callback checking if the gradient call count surpasses the value grad_limit.
If so, sets the stop_criterion to true 
"""
function f_call_callback(f_call_limit)
    return function limit_check!(state) 
        return state.f.counter < f_call_limit
    end
end

"""
    grad_call_callback(grad_call_limit)

Callback checking if the gradient call count surpasses the value grad_limit.
If so, sets the stop_criterion to true 
"""
function grad_call_callback(grad_call_limit)
    return function limit_check!(state) 
        return state.grad.counter < grad_call_limit
    end
end

"""
    lmo_call_callback(lmo_call_limit)

Callback checking if the lmo call count surpasses the value lmo_limit.
If so, sets the stop_criterion to true 
"""
function lmo_call_callback(lmo_call_limit)
    return function limit_check!(state) 
        return state.lmo.counter < lmo_call_limit
    end
end