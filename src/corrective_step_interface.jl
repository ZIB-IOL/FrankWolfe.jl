
abstract type CorrectiveStep end

"""
    run_corrective_step(corrective_step, f, grad!, gradient, x, active_set, t, lmo, line_search, linesearch_workspace, primal, phi, tot_time, callback, renorm_interval) -> (x, phi, primal)

Corrective step method specific to the `CS` corrective_step type.
The corrective step can perform whatever update over the current active set, the function should return the new iterate  a FW step should be run next with the boolean `should_fw_step` and compute a new dual gap estimate `phi`.
"""
function run_corrective_step end

"""
    prepare_corrective_step(corrective_step::CS, f, grad!, gradient, active_set, t, lmo, primal, phi, tot_time) -> (should_compute_vertex, use_corrective)

`should_compute_vertex` is a boolean flag deciding whether a new vertex should be computed.
`use_corrective` is a function that takes the vertex (the vertex is a valid new vertex only if should_compute_vertex was true)
"""
function prepare_corrective_step end

"""
    (Lazified) away-step for corrective Frank-Wolfe
"""
struct AwayStep <: CorrectiveStep
    lazy::Bool
end

AwayStep() = AwayStep(false)

function prepare_corrective_step(corrective_step::AwayStep, f, grad!, gradient, active_set, t, lmo, primal, phi)
    should_compute_vertex = !corrective_step.lazy
    return should_compute_vertex
end

function run_corrective_step(corrective_step::AwayStep, f, grad!, gradient, x, v, dual_gap, active_set, t, lmo, line_search, linesearch_workspace, primal, phi, tot_time, callback, renorm_interval, memory_mode, epsilon, lazy_tolerance, d)
    _, v_lazy, v_loc, _, a_lambda, a, a_loc, _, _ = active_set_argminmax(active_set, gradient)
    grad_dot_x = fast_dot(x, gradient)
    grad_dot_a = fast_dot(a, gradient)
    away_gap = grad_dot_a - grad_dot_x
    new_phi = phi
    # flag for whether callback interrupts the solving process
    should_continue = true
    if !corrective_step.lazy
        if away_gap >= dual_gap
            gamma_max = a_lambda / (1 - a_lambda)
            d = muladd_memory_mode(memory_mode, d, a, x)
            gamma = perform_line_search(
                line_search,
                t,
                f,
                grad!,
                gradient,
                x,
                d,
                gamma_max,
                linesearch_workspace,
                memory_mode,
            )
            gamma = min(gamma_max, gamma)
            step_type = gamma ≈ gamma_max ? ST_DROP : ST_AWAY
            should_fw_step = false
            callback_state = CallbackState(
                t,
                primal,
                primal - new_phi,
                new_phi,
                tot_time,
                x,
                v,
                d,
                gamma,
                f,
                grad!,
                lmo,
                gradient,
                step_type,
            )
            if callback !== nothing
                should_continue = callback(callback_state)
            end
            active_set_update!(active_set, -gamma, a, true, a_loc)        
        else
            should_fw_step = true
        end
    else # lazy AFW
        # compute the local FW gap over the active set 
        away_step_taken = false
        fw_step_taken = false
        grad_dot_lazy_fw_vertex = fast_dot(v_lazy, gradient)
        lazy_gap = grad_dot_x - grad_dot_lazy_fw_vertex
        if lazy_gap >= max(away_gap, phi / lazy_tolerance, epsilon)
            step_type = ST_LAZY
            gamma_max = one(a_lambda)
            d = muladd_memory_mode(memory_mode, d, x, v_lazy)
            vertex = v
            fw_step_taken = true
            index = v_loc
            should_fw_step = false
        elseif away_gap >= max(phi / lazy_tolerance, epsilon)
            step_type = ST_AWAY
            gamma_max = a_lambda / (1 - a_lambda)
            d = muladd_memory_mode(memory_mode, d, a, x)
            vertex = a
            away_step_taken = true
            index = a_loc
            should_fw_step = false
        else
            # call the true LMO since `v` was not updated
            step_type = ST_REGULAR
            v = compute_extreme_point(lmo, gradient)
            grad_dot_fw_vertex = fast_dot(v, gradient)
            dual_gap = grad_dot_x - grad_dot_fw_vertex
            should_fw_step = true
        end
        if fw_step_taken || away_step_taken
            gamma = perform_line_search(
                line_search,
                t,
                f,
                grad!,
                gradient,
                x,
                d,
                gamma_max,
                linesearch_workspace,
                memory_mode,
            )
            gamma = min(gamma_max, gamma)
            step_type = gamma ≈ gamma_max ? ST_DROP : step_type
            # cleanup and renormalize every x iterations. Only for the fw steps.
            renorm = mod(t, renorm_interval) == 0
            if away_step_taken
                active_set_update!(active_set, -gamma, vertex, true, index)
            else
                active_set_update!(active_set, gamma, vertex, renorm, index)
            end
        end
    end
    return x, v, phi, dual_gap, should_fw_step, should_continue
end
