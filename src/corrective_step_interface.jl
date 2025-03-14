
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
            if callback !== nothing
                state = CallbackState(
                    t,
                    primal,
                    primal - phi,
                    phi,
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
                should_continue = callback(state, active_set)
            end
            active_set_update!(active_set, -gamma, a, true, a_loc)        
        else
            should_fw_step = true
        end
    else # lazy AFW
        # compute the local FW gap over the active set 
        away_step_taken = false
        lazy_fw_step_taken = false
        grad_dot_lazy_fw_vertex = fast_dot(v_lazy, gradient)
        lazy_gap = grad_dot_x - grad_dot_lazy_fw_vertex
        if lazy_gap >= max(away_gap, phi / lazy_tolerance, epsilon)
            step_type = ST_LAZY
            gamma_max = one(a_lambda)
            d = muladd_memory_mode(memory_mode, d, x, v_lazy)
            vertex = v_lazy
            lazy_fw_step_taken = true
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
            v = vertex = compute_extreme_point(lmo, gradient)
            grad_dot_fw_vertex = fast_dot(v, gradient)
            dual_gap = grad_dot_x - grad_dot_fw_vertex
            # if enough progress, perform regular FW step
            if dual_gap >= phi / lazy_tolerance
                should_fw_step = true
            else
                step_type = ST_DUALSTEP
                phi = min(dual_gap, phi / 2)
                should_fw_step = false
                state = CallbackState(
                    t,
                    primal,
                    primal - phi,
                    phi,
                    tot_time,
                    x,
                    v,
                    d,
                    zero(a_lambda),
                    f,
                    grad!,
                    lmo,
                    gradient,
                    step_type,
                )
                if callback !== nothing
                    should_continue = callback(state, active_set)
                end
            end
        end
        if lazy_fw_step_taken || away_step_taken
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
            state = CallbackState(
                t,
                primal,
                primal - phi,
                phi,
                tot_time,
                x,
                vertex,
                d,
                gamma,
                f,
                grad!,
                lmo,
                gradient,
                step_type,
            )
            if callback !== nothing
                should_continue = callback(state, active_set)
            end
            # cleanup and renormalize every x iterations. Only for the fw steps.
            renorm = mod(t, renorm_interval) == 0
            if away_step_taken
                active_set_update!(active_set, -gamma, vertex, renorm, index)
            else
                active_set_update!(active_set, gamma, vertex, renorm, index)
            end
            if mod(t, renorm_interval) == 0
                active_set_renormalize!(active_set)
                x = compute_active_set_iterate!(active_set)
            end
        end
    end
    return x, v, phi, dual_gap, should_fw_step, should_continue
end

struct BlendedPairwiseStep <: CorrectiveStep
    lazy::Bool
end

BlendedPairwiseStep() = BlendedPairwiseStep(false)

function prepare_corrective_step(corrective_step::BlendedPairwiseStep, f, grad!, gradient, active_set, t, lmo, primal, phi)
    should_compute_vertex = !corrective_step.lazy
    return should_compute_vertex
end

function run_corrective_step(corrective_step::BlendedPairwiseStep, f, grad!, gradient, x, v, dual_gap, active_set, t, lmo, line_search, linesearch_workspace, primal, phi, tot_time, callback, renorm_interval, memory_mode, epsilon, lazy_tolerance, d)
    _, v_local, v_loc, _, a_lambda, a, a_loc, _, _ = active_set_argminmax(active_set, gradient)
    grad_dot_x = fast_dot(x, gradient)
    grad_dot_a = fast_dot(a, gradient)
    grad_dot_local_fw_vertex = fast_dot(v_local, gradient)
    local_gap = grad_dot_a - grad_dot_local_fw_vertex
    # flag for whether callback interrupts the solving process
    should_continue = true
    # perform local step if the local_gap promises enough progress
    # if nonlazy, phi is already computed as the true dual gap
    if local_gap >= max(phi / lazy_tolerance, epsilon)
        d = muladd_memory_mode(memory_mode, d, a, v_local)
        vertex_taken = v_local
        gamma_max = a_lambda
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
        step_type = gamma ≈ gamma_max ? ST_DROP : ST_PAIRWISE
        should_fw_step = false
        state = CallbackState(
            t,
            primal,
            primal - phi,
            phi,
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
            should_continue = callback(state, active_set)
        end
        active_set_update_pairwise!(
            active_set,
            gamma,
            gamma_max,
            v_loc,
            a_loc,
            vertex_taken,
            a,
            false,
            nothing,
        )
    else # perform normal FW step
        if !corrective_step.lazy
            # v computed above already
            should_fw_step = true
        else # lazy case, v needs to be computed here
            v = compute_extreme_point(lmo, gradient)
            dual_gap = grad_dot_x - fast_dot(gradient, v)
            # FW vertex promises progress
            if dual_gap ≥ max(epsilon, phi / lazy_tolerance)
                should_fw_step = true
            else
                should_fw_step = false
                step_type = ST_DUALSTEP
                phi = min(dual_gap, phi / 2)
                if callback !== nothing
                    gamma = zero(a_lambda)
                    state = CallbackState(
                        t,
                        primal,
                        primal - phi,
                        phi,
                        tot_time,
                        x,
                        v,
                        nothing,
                        gamma,
                        f,
                        grad!,
                        lmo,
                        gradient,
                        step_type,
                    )
                    should_continue = callback(state, active_set)
                end
            end
        end
    end
    return x, v, phi, dual_gap, should_fw_step, should_continue
end

"""
Compares a pairwise and away step and chooses the one with most progress.
The line search is computed for both steps.
If one step incurs a drop, it is favored, otherwise the one decreasing the primal value the most is favored.
"""
struct HybridPairAwayStep{DT} <: CorrectiveStep
    lazy::Bool
    d_pairwise::DT
end


function prepare_corrective_step(corrective_step::HybridPairAwayStep, f, grad!, gradient, active_set, t, lmo, primal, phi)
    return !corrective_step.lazy
end

function run_corrective_step(corrective_step::HybridPairAwayStep, f, grad!, gradient, x, v, dual_gap, active_set, t, lmo, line_search, linesearch_workspace, primal, phi, tot_time, callback, renorm_interval, memory_mode, epsilon, lazy_tolerance, d)
    _, v_local, v_loc, _, a_lambda, a, a_loc, _, _ = active_set_argminmax(active_set, gradient)
    grad_dot_x = fast_dot(x, gradient)
    grad_dot_a = fast_dot(a, gradient)
    grad_dot_local_fw_vertex = fast_dot(v_local, gradient)
    pairwise_gap = grad_dot_a - grad_dot_local_fw_vertex
    lazy_gap = grad_dot_x - grad_dot_local_fw_vertex
    # flag for whether callback interrupts the solving process
    should_continue = true
    # if not enough progress from pairwise or local, directly perform a FW step
    if max(pairwise_gap, lazy_gap) < max(phi / lazy_tolerance, epsilon)
        if !corrective_step.lazy
            # v computed above already
            should_fw_step = true
        else # lazy case, v needs to be computed here
            v = compute_extreme_point(lmo, gradient)
            dual_gap = grad_dot_x - fast_dot(gradient, v)
            # FW vertex promises progress
            if dual_gap ≥ max(epsilon, phi / lazy_tolerance)
                should_fw_step = true
            else
                should_fw_step = false
                step_type = ST_DUALSTEP
                phi = min(dual_gap, phi / 2)
                if callback !== nothing
                    gamma = zero(a_lambda)
                    state = CallbackState(
                        t,
                        primal,
                        primal - phi,
                        phi,
                        tot_time,
                        x,
                        v,
                        nothing,
                        gamma,
                        f,
                        grad!,
                        lmo,
                        gradient,
                        step_type,
                    )
                    should_continue = callback(state, active_set)
                end
            end
        end
    elseif pairwise_gap > max(phi / lazy_tolerance, epsilon)
        should_fw_step = false
        d_pairwise = muladd_memory_mode(memory_mode, corrective_step.d_pairwise, a, v_local)
        vertex_taken = v_local
        gamma_max_pairiwse = a_lambda
        gamma_pairwise = perform_line_search(
            line_search,
            t,
            f,
            grad!,
            gradient,
            x,
            d_pairwise,
            gamma_max_pairiwse,
            linesearch_workspace,
            memory_mode,
        )
        gamma_pairwise = min(gamma_max_pairiwse, gamma_pairwise)
        step_type_pairwise = gamma_pairwise ≈ gamma_max_pairiwse ? ST_DROP : ST_PAIRWISE

        d_away = muladd_memory_mode(memory_mode, d, a, x)
        vertex_taken = v_local
        gamma_max_away = a_lambda / (1 - a_lambda)
        gamma_away = perform_line_search(
            line_search,
            t,
            f,
            grad!,
            gradient,
            x,
            d_away,
            gamma_max_away,
            linesearch_workspace,
            memory_mode,
        )
        gamma_away = min(gamma_max_away, gamma_away)
        step_type_away = gamma_away ≈ gamma_max_away ? ST_DROP : ST_AWAY

        select_away = false
        # both drop, take the most primal progress
        if step_type_away == ST_DROP && step_type_pairwise == ST_DROP
            if f(x - gamma_away * d_away) < f(x - gamma_pairwise * d_pairwise)
                select_away = true
            end
        elseif step_type_away == ST_DROP
            select_away = true
        elseif step_type_pairwise == ST_DROP
            select_away = false
        else # none drops, take the most primal progress
            if f(x - gamma_away * d_away) < f(x - gamma_pairwise * d_pairwise)
                select_away = true
            end
        end
        if select_away
            d = d_away
            gamma = gamma_away
            step_type = step_type_away
        else
            d = d_pairwise
            gamma = gamma_pairwise
            step_type = step_type_pairwise
        end
        
        state = CallbackState(
            t,
            primal,
            primal - phi,
            phi,
            tot_time,
            x,
            v_local,
            d,
            gamma,
            f,
            grad!,
            lmo,
            gradient,
            step_type,
        )
        if callback !== nothing
            should_continue = callback(state, active_set)
        end
        # cleanup and renormalize every x iterations. Only for the fw steps.
        renorm = mod(t, renorm_interval) == 0
        if select_away
            active_set_update!(active_set, -gamma, a, renorm, index)
        else
            active_set_update_pairwise!(
                active_set,
                gamma,
                gamma_max,
                v_loc,
                a_loc,
                v_local,
                a,
                false,
                nothing,
            )
        end
        state = CallbackState(
            t,
            primal,
            primal - phi,
            phi,
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
            should_continue = callback(state, active_set)
        end
        if mod(t, renorm_interval) == 0
            active_set_renormalize!(active_set)
            x = compute_active_set_iterate!(active_set)
        end
    else # perform local step if one of the local gaps promises enough progress
        step_type = ST_LAZY
        gamma_max = one(a_lambda)
        d = muladd_memory_mode(memory_mode, d, x, v_lazy)
        vertex = v_lazy
        index = v_loc

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
        state = CallbackState(
            t,
            primal,
            primal - phi,
            phi,
            tot_time,
            x,
            vertex,
            d,
            gamma,
            f,
            grad!,
            lmo,
            gradient,
            step_type,
        )
        if callback !== nothing
            should_continue = callback(state, active_set)
        end
        renorm = mod(t, renorm_interval) == 0
        active_set_update!(active_set, gamma, vertex, renorm, index)
        if mod(t, renorm_interval) == 0
            active_set_renormalize!(active_set)
            x = compute_active_set_iterate!(active_set)
        end
    end
    return x, v, phi, dual_gap, should_fw_step, should_continue
end
