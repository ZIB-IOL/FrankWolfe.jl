
abstract type CorrectiveStep end

"""
    run_corrective_step(corrective_step, f, grad!, gradient, x, active_set, t, lmo, line_search, linesearch_workspace, primal, phi_value, tot_time, callback, renorm_interval) -> (x, phi_value, primal)

Corrective step method specific to the `CS` corrective_step type.
The corrective step can perform whatever update over the current active set, the function should return the new iterate  a FW step should be run next with the boolean `should_fw_step` and compute a new dual gap estimate `phi_value`.
"""
function run_corrective_step end

"""
    prepare_corrective_step(corrective_step::CS, f, grad!, gradient, active_set, t, lmo, primal, phi_value, tot_time) -> (should_compute_vertex, use_corrective)

`should_compute_vertex` is a boolean flag deciding whether a new vertex should be computed.
`use_corrective` is a function that takes the vertex (the vertex is a valid new vertex only if should_compute_vertex was true)
"""
function prepare_corrective_step end

"""
    (Lazified) away-step for corrective Frank-Wolfe
"""
struct AwayStep{T} <: CorrectiveStep
    lazy::Bool
    lazy_tolerance::T
end

AwayStep(lazy=false) = AwayStep(lazy, 2.0)

function prepare_corrective_step(
    corrective_step::AwayStep,
    f,
    grad!,
    gradient,
    active_set,
    t,
    lmo,
    primal,
    phi_value,
)
    should_compute_vertex = !corrective_step.lazy
    return should_compute_vertex
end

function run_corrective_step(
    corrective_step::AwayStep,
    f,
    grad!,
    gradient,
    x,
    v,
    dual_gap,
    active_set,
    t,
    lmo,
    line_search,
    linesearch_workspace,
    primal,
    phi_value,
    tot_time,
    callback,
    renorm_interval,
    memory_mode,
    epsilon,
    d,
)
    _, v_lazy, v_loc, _, a_lambda, a, a_loc, _, _ = active_set_argminmax(active_set, gradient)
    grad_dot_x = dot(gradient, x)
    grad_dot_a = dot(gradient, a)
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
                    primal - phi_value,
                    phi_value,
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
        grad_dot_lazy_fw_vertex = dot(gradient, v_lazy)
        lazy_gap = grad_dot_x - grad_dot_lazy_fw_vertex
        if lazy_gap >= max(away_gap, phi_value / corrective_step.lazy_tolerance, epsilon)
            step_type = ST_LAZY
            gamma_max = one(a_lambda)
            d = muladd_memory_mode(memory_mode, d, x, v_lazy)
            vertex = v_lazy
            lazy_fw_step_taken = true
            index = v_loc
            should_fw_step = false
        elseif away_gap >= max(phi_value / corrective_step.lazy_tolerance, epsilon)
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
            grad_dot_fw_vertex = dot(gradient, v)
            dual_gap = grad_dot_x - grad_dot_fw_vertex
            # if enough progress, perform regular FW step
            if dual_gap >= phi_value / corrective_step.lazy_tolerance
                should_fw_step = true
            else
                step_type = ST_DUALSTEP
                phi_value = min(dual_gap, phi_value / 2)
                should_fw_step = false
                state = CallbackState(
                    t,
                    primal,
                    primal - phi_value,
                    phi_value,
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
                primal - phi_value,
                phi_value,
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
    return x, v, phi_value, dual_gap, should_fw_step, should_continue
end

struct BlendedPairwiseStep{T} <: CorrectiveStep
    lazy::Bool
    lazy_tolerance::T
end

BlendedPairwiseStep(lazy=false) = BlendedPairwiseStep(lazy, 2.0)

function prepare_corrective_step(
    corrective_step::BlendedPairwiseStep,
    f,
    grad!,
    gradient,
    active_set,
    t,
    lmo,
    primal,
    phi_value,
)
    should_compute_vertex = !corrective_step.lazy
    return should_compute_vertex
end

function run_corrective_step(
    corrective_step::BlendedPairwiseStep,
    f,
    grad!,
    gradient,
    x,
    v,
    dual_gap,
    active_set,
    t,
    lmo,
    line_search,
    linesearch_workspace,
    primal,
    phi_value,
    tot_time,
    callback,
    renorm_interval,
    memory_mode,
    epsilon,
    d,
)
    _, v_local, v_loc, _, a_lambda, a, a_loc, _, _ = active_set_argminmax(active_set, gradient)
    grad_dot_x = dot(gradient, x)
    grad_dot_a = dot(gradient, a)
    grad_dot_local_fw_vertex = dot(gradient, v_local)
    local_gap = grad_dot_a - grad_dot_local_fw_vertex
    # flag for whether callback interrupts the solving process
    should_continue = true
    # perform local step if the local_gap promises enough progress
    # if nonlazy, phi_value is already computed as the true dual gap
    if local_gap >= max(phi_value / corrective_step.lazy_tolerance, epsilon)
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
            primal - phi_value,
            phi_value,
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
            dual_gap = grad_dot_x - dot(gradient, v)
            # FW vertex promises progress
            if dual_gap ≥ max(epsilon, phi_value / corrective_step.lazy_tolerance)
                should_fw_step = true
            else
                should_fw_step = false
                step_type = ST_DUALSTEP
                phi_value = min(dual_gap, phi_value / 2)
                if callback !== nothing
                    gamma = zero(a_lambda)
                    state = CallbackState(
                        t,
                        primal,
                        primal - phi_value,
                        phi_value,
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
    return x, v, phi_value, dual_gap, should_fw_step, should_continue
end

"""
Compares a pairwise and away step and chooses the one with most progress.
The line search is computed for both steps.
If one step incurs a drop, it is favored, otherwise the one decreasing the primal value the most is favored.
"""
struct HybridPairAwayStep{DT,T} <: CorrectiveStep
    lazy::Bool
    d_pairwise::DT
    lazy_tolerance::T
end

HybridPairAwayStep(lazy, d_pairwise) = HybridPairAwayStep(lazy, d_pairwise, 2.0)

function prepare_corrective_step(
    corrective_step::HybridPairAwayStep,
    f,
    grad!,
    gradient,
    active_set,
    t,
    lmo,
    primal,
    phi_value,
)
    return !corrective_step.lazy
end

function run_corrective_step(
    corrective_step::HybridPairAwayStep,
    f,
    grad!,
    gradient,
    x,
    v,
    dual_gap,
    active_set,
    t,
    lmo,
    line_search,
    linesearch_workspace,
    primal,
    phi_value,
    tot_time,
    callback,
    renorm_interval,
    memory_mode,
    epsilon,
    d,
)
    _, v_local, v_loc, _, a_lambda, a, a_loc, _, _ = active_set_argminmax(active_set, gradient)
    grad_dot_x = dot(gradient, x)
    grad_dot_a = dot(gradient, a)
    grad_dot_local_fw_vertex = dot(gradient, v_local)
    pairwise_gap = grad_dot_a - grad_dot_local_fw_vertex
    lazy_gap = grad_dot_x - grad_dot_local_fw_vertex
    # flag for whether callback interrupts the solving process
    should_continue = true
    # if not enough progress from pairwise or local, directly perform a FW step
    if max(pairwise_gap, lazy_gap) < max(phi_value / corrective_step.lazy_tolerance, epsilon)
        if !corrective_step.lazy
            # v computed above already
            should_fw_step = true
        else # lazy case, v needs to be computed here
            v = compute_extreme_point(lmo, gradient)
            dual_gap = grad_dot_x - dot(gradient, v)
            # FW vertex promises progress
            if dual_gap ≥ max(epsilon, phi_value / corrective_step.lazy_tolerance)
                should_fw_step = true
            else
                should_fw_step = false
                step_type = ST_DUALSTEP
                phi_value = min(dual_gap, phi_value / 2)
                if callback !== nothing
                    gamma = zero(a_lambda)
                    state = CallbackState(
                        t,
                        primal,
                        primal - phi_value,
                        phi_value,
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
    elseif pairwise_gap > max(phi_value / corrective_step.lazy_tolerance, epsilon)
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
            primal - phi_value,
            phi_value,
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
            active_set_update!(active_set, -gamma, a, renorm, a_loc)
        else
            active_set_update_pairwise!(
                active_set,
                gamma,
                gamma_max_pairiwse,
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
            primal - phi_value,
            phi_value,
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
            primal - phi_value,
            phi_value,
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
        active_set_update!(active_set, gamma, vertex, renorm, v_loc)
        if mod(t, renorm_interval) == 0
            active_set_renormalize!(active_set)
            x = compute_active_set_iterate!(active_set)
        end
    end
    return x, v, phi_value, dual_gap, should_fw_step, should_continue
end

"""
Computes a classic pairwise step, i.e., `d = v^FW - v^away`.
"""
struct PairwiseStep{T} <: CorrectiveStep
    lazy::Bool
    lazy_tolerance::T
end

PairwiseStep(lazy=false, lazy_tolerance=2.0) = PairwiseStep(lazy, lazy_tolerance)

function prepare_corrective_step(
    corrective_step::PairwiseStep,
    f,
    grad!,
    gradient,
    active_set,
    t,
    lmo,
    primal,
    phi_value,
)
    return !corrective_step.lazy
end

function run_corrective_step(
    corrective_step::PairwiseStep,
    f,
    grad!,
    gradient,
    x,
    v,
    dual_gap,
    active_set,
    t,
    lmo,
    line_search,
    linesearch_workspace,
    primal,
    phi_value,
    tot_time,
    callback,
    renorm_interval,
    memory_mode,
    epsilon,
    d,
)
    _, v_local, v_loc, _, a_lambda, a, a_loc, _, _ = active_set_argminmax(active_set, gradient)
    grad_dot_x = dot(gradient, x)
    grad_dot_a = dot(gradient, a)
    grad_dot_local_fw_vertex = dot(gradient, v_local)
    local_pairwise_gap = grad_dot_a - grad_dot_local_fw_vertex
    # flag for whether callback interrupts the solving process
    should_continue = true
    should_fw_step = false
    take_local = false
    fw_index = v_loc
    if corrective_step.lazy &&
       local_pairwise_gap >= max(phi_value / corrective_step.lazy_tolerance, epsilon)
        fw_vertex = v_local
        d = muladd_memory_mode(memory_mode, d, a, v_local)
        take_local = true
    else
        # if not lazy, v is already computed
        if corrective_step.lazy
            v = compute_extreme_point(lmo, gradient)
            dual_gap = grad_dot_x - dot(gradient, v)
            phi_value = min(phi_value, dual_gap)
        end
        d = muladd_memory_mode(memory_mode, d, a, v)
        fw_vertex = v
    end
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
    step_type = if gamma ≈ gamma_max
        ST_DROP
    elseif take_local
        ST_LAZY
    else
        ST_PAIRWISE
    end
    state = CallbackState(
        t,
        primal,
        primal - phi_value,
        phi_value,
        tot_time,
        x,
        fw_vertex,
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
    # away update
    active_set_update!(active_set, -gamma, a, false, a_loc)
    # fw update
    fw_index = take_local ? v_loc : nothing
    active_set_update!(active_set, gamma, fw_vertex, true, fw_index)
    return x, v, phi_value, dual_gap, should_fw_step, should_continue
end

# Shared fallback when the local strong-Wolfe gap is too small for a hull step:
# non-lazy → FW vertex already computed; lazy → LMO, then FW or dual (gap) step.
function _corrective_fw_or_dual_step(
    lazy,
    lazy_tolerance,
    gradient,
    x,
    v,
    dual_gap,
    phi_value,
    lmo,
    epsilon,
    callback,
    t,
    primal,
    tot_time,
    f,
    grad!,
    active_set,
)
    should_continue = true
    if !lazy
        return v, dual_gap, phi_value, true, should_continue
    end
    v = compute_extreme_point(lmo, gradient)
    dual_gap = dot(gradient, x) - dot(gradient, v)
    if dual_gap ≥ max(epsilon, phi_value / lazy_tolerance)
        return v, dual_gap, phi_value, true, should_continue
    end
    phi_value = min(dual_gap, phi_value / 2)
    if callback !== nothing
        state = CallbackState(
            t,
            primal,
            primal - phi_value,
            phi_value,
            tot_time,
            x,
            v,
            nothing,
            zero(eltype(x)),
            f,
            grad!,
            lmo,
            gradient,
            ST_DUALSTEP,
        )
        should_continue = callback(state, active_set)
    end
    return v, dual_gap, phi_value, false, should_continue
end

# Euclidean projection onto the probability simplex. Dropped coordinates are the
# tail of the descending sort after the first `n_keep` entries.
function _simplex_projection_with_drops(x; s=one(eltype(x)))
    n = length(x)
    v = x .- maximum(x)
    perm = sortperm(v; rev=true)
    u = v[perm]
    cssv = cumsum(u)
    n_keep = count(j -> u[j] * j > cssv[j] - s, eachindex(u))
    n_keep = max(n_keep, 1)
    theta = (cssv[n_keep] - s) / n_keep
    w = zeros(eltype(v), n)
    @inbounds for i in 1:n_keep
        w[perm[i]] = v[perm[i]] - theta
    end
    drop_indices = n_keep == n ? Int[] : sort(perm[(n_keep+1):n])
    return w, drop_indices
end

_linesearch_tol(ls) = hasproperty(ls, :tol) ? ls.tol : 1e-8

"""
    SimplexGradientDescentStep(lazy=false; lazy_tolerance=2.0, line_search_inner=Secant())

One simplex gradient descent (SiGD) step over the active set, as in
Braun et al., "Blended Conditional Gradients", Algorithm 2.

If the local strong-Wolfe gap over the active set is large enough compared to
`phi_value / lazy_tolerance`, a single SiGD step is taken. Otherwise a Frank-Wolfe
step is requested from [`corrective_frank_wolfe`](@ref).
"""
mutable struct SimplexGradientDescentStep{T,LS} <: CorrectiveStep
    lazy::Bool
    lazy_tolerance::T
    line_search_inner::LS
    linesearch_inner_workspace::Any
end

function SimplexGradientDescentStep(lazy=false; lazy_tolerance=2.0, line_search_inner=Secant())
    return SimplexGradientDescentStep(lazy, lazy_tolerance, line_search_inner, nothing)
end

function prepare_corrective_step(
    corrective_step::SimplexGradientDescentStep,
    f,
    grad!,
    gradient,
    active_set,
    t,
    lmo,
    primal,
    phi_value,
)
    return !corrective_step.lazy
end

function run_corrective_step(
    corrective_step::SimplexGradientDescentStep,
    f,
    grad!,
    gradient,
    x,
    v,
    dual_gap,
    active_set,
    t,
    lmo,
    line_search,
    linesearch_workspace,
    primal,
    phi_value,
    tot_time,
    callback,
    renorm_interval,
    memory_mode,
    epsilon,
    d,
)
    _, v_local, _, _, _, a, _, _, _ = active_set_argminmax(active_set, gradient)
    local_gap = dot(gradient, a) - dot(gradient, v_local)
    should_continue = true

    if local_gap < max(phi_value / corrective_step.lazy_tolerance, epsilon)
        v, dual_gap, phi_value, should_fw_step, should_continue = _corrective_fw_or_dual_step(
            corrective_step.lazy,
            corrective_step.lazy_tolerance,
            gradient,
            x,
            v,
            dual_gap,
            phi_value,
            lmo,
            epsilon,
            callback,
            t,
            primal,
            tot_time,
            f,
            grad!,
            active_set,
        )
        return x, v, phi_value, dual_gap, should_fw_step, should_continue
    end

    if corrective_step.linesearch_inner_workspace === nothing
        corrective_step.linesearch_inner_workspace =
            build_linesearch_workspace(corrective_step.line_search_inner, x, gradient)
    end
    line_search_inner = corrective_step.line_search_inner
    if line_search_inner isa Adaptive
        line_search_inner.L_est = Inf
    end

    c = [dot(gradient, atom) for atom in active_set.atoms]
    k = length(active_set)
    csum = sum(c)
    c .-= (csum / k)
    dir = c
    ls_tol = _linesearch_tol(line_search_inner)
    descent_direction_product = dot(dir, dir) + (csum / k) * sum(dir)

    # Check if the descent direction is too small, if so, proceed to the FW or dual step
    if descent_direction_product < ls_tol
        bdir = big.(gradient)
        c = [dot(bdir, atom) for atom in active_set.atoms]
        csum = sum(c)
        c .-= csum / k
        dir = c
        descent_direction_product = dot(dir, dir) + (csum / k) * sum(dir)
        if descent_direction_product < ls_tol
            v, dual_gap, phi_value, should_fw_step, should_continue = _corrective_fw_or_dual_step(
                corrective_step.lazy,
                corrective_step.lazy_tolerance,
                gradient,
                x,
                v,
                dual_gap,
                phi_value,
                lmo,
                epsilon,
                callback,
                t,
                primal,
                tot_time,
                f,
                grad!,
                active_set,
            )
            return x, v, phi_value, dual_gap, should_fw_step, should_continue
        end
    end

    # Ratio test: largest η ≥ 0 such that λ - η dir ≥ 0. The coordinates that
    # hit zero first are the ones dropped on a drop step.
    η = eltype(dir)(Inf)
    drop_indices = Int[]
    purge_tol = 2 * weight_purge_threshold_default(eltype(active_set.weights))
    @inbounds for idx in eachindex(dir)
        if dir[idx] > 0
            η_idx = active_set.weights[idx] / dir[idx]
            if abs(η_idx - η) ≤ purge_tol
                push!(drop_indices, idx)
            elseif η_idx < η
                η = η_idx
                empty!(drop_indices)
                push!(drop_indices, idx)
            end
        end
    end
    η = isfinite(η) ? max(zero(η), η) : zero(eltype(dir))
    λ_boundary = active_set.weights .- η .* dir
    for idx in drop_indices
        λ_boundary[idx] = zero(eltype(λ_boundary))
    end
    x_prev = copy(active_set.x)
    y = similar(x_prev)
    y .= 0
    for (λi, ai) in zip(λ_boundary, active_set.atoms)
        @. y += λi * ai
    end
    gamma = one(η)

    # If the drop is non-increasing, perform a drop step
    if f(x_prev) ≥ f(y)
        step_type = ST_DROP
        update_weights!(active_set, λ_boundary)
        deleteat!(active_set, drop_indices)
        compute_active_set_iterate!(active_set)

        # Otherwise, perform a simplex descent step with line search
    else
        d = muladd_memory_mode(memory_mode, d, x_prev, y)
        if line_search_inner isa Adaptive
            gamma = perform_line_search(
                line_search_inner,
                t,
                f,
                grad!,
                gradient,
                x_prev,
                d,
                one(eltype(x_prev)),
                corrective_step.linesearch_inner_workspace,
                memory_mode,
            )
            if gamma < eps(float(gamma))
                gamma = perform_line_search(
                    line_search_inner,
                    t,
                    f,
                    grad!,
                    gradient,
                    x_prev,
                    d,
                    one(eltype(x_prev)),
                    corrective_step.linesearch_inner_workspace,
                    memory_mode,
                    should_upgrade=Val{true}(),
                )
            end
        else
            if dot(gradient, x_prev - y) < ls_tol
                v, dual_gap, phi_value, should_fw_step, should_continue =
                    _corrective_fw_or_dual_step(
                        corrective_step.lazy,
                        corrective_step.lazy_tolerance,
                        gradient,
                        x,
                        v,
                        dual_gap,
                        phi_value,
                        lmo,
                        epsilon,
                        callback,
                        t,
                        primal,
                        tot_time,
                        f,
                        grad!,
                        active_set,
                    )
                return x, v, phi_value, dual_gap, should_fw_step, should_continue
            end
            gamma = perform_line_search(
                line_search_inner,
                t,
                f,
                grad!,
                gradient,
                x_prev,
                d,
                one(eltype(x_prev)),
                corrective_step.linesearch_inner_workspace,
                memory_mode,
            )
        end
        gamma = min(one(gamma), gamma)
        if gamma == one(gamma)
            step_type = ST_DROP
            update_weights!(active_set, λ_boundary)
            deleteat!(active_set, drop_indices)
            compute_active_set_iterate!(active_set)
        else
            step_type = ST_SIMPLEXDESCENT
            update_weights!(active_set, active_set.weights .- gamma * η .* dir)
            compute_active_set_iterate!(active_set)
        end
    end

    x = get_active_set_iterate(active_set)
    d = muladd_memory_mode(memory_mode, d, x_prev, y)
    if callback !== nothing
        state = CallbackState(
            t,
            primal,
            primal - phi_value,
            phi_value,
            tot_time,
            x_prev,
            y,
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
    if mod(t, renorm_interval) == 0
        active_set_renormalize!(active_set)
        x = compute_active_set_iterate!(active_set)
    end
    return x, v, phi_value, dual_gap, false, should_continue
end

"""
    ProjectedGradientDescentStep(; hessian, lazy=false, lazy_tolerance=2.0, accelerated=false)

One projected gradient (or Nesterov-accelerated) step over the probability simplex
of barycentric coordinates of the active set.

Requires a Hessian of `f` to build the reduced quadratic and to estimate `L`
(and `μ` if `accelerated=true`). If the local strong-Wolfe gap is small, a
Frank-Wolfe step is requested from [`corrective_frank_wolfe`](@ref).
"""
mutable struct ProjectedGradientDescentStep{H,T} <: CorrectiveStep
    lazy::Bool
    lazy_tolerance::T
    hessian::H
    accelerated::Bool
    y::Any
    alpha::Float64
end

function ProjectedGradientDescentStep(; hessian, lazy=false, lazy_tolerance=2.0, accelerated=false)
    hessian === nothing && throw(ArgumentError("ProjectedGradientDescentStep requires a hessian"))
    return ProjectedGradientDescentStep(lazy, lazy_tolerance, hessian, accelerated, nothing, 0.0)
end

function prepare_corrective_step(
    corrective_step::ProjectedGradientDescentStep,
    f,
    grad!,
    gradient,
    active_set,
    t,
    lmo,
    primal,
    phi_value,
)
    return !corrective_step.lazy
end

function run_corrective_step(
    corrective_step::ProjectedGradientDescentStep,
    f,
    grad!,
    gradient,
    x,
    v,
    dual_gap,
    active_set,
    t,
    lmo,
    line_search,
    linesearch_workspace,
    primal,
    phi_value,
    tot_time,
    callback,
    renorm_interval,
    memory_mode,
    epsilon,
    d,
)
    _, v_local, _, _, _, a, _, _, _ = active_set_argminmax(active_set, gradient)
    local_gap = dot(gradient, a) - dot(gradient, v_local)
    should_continue = true
    progress_threshold = max(phi_value / corrective_step.lazy_tolerance, epsilon)

    if local_gap < progress_threshold
        v, dual_gap, phi_value, should_fw_step, should_continue = _corrective_fw_or_dual_step(
            corrective_step.lazy,
            corrective_step.lazy_tolerance,
            gradient,
            x,
            v,
            dual_gap,
            phi_value,
            lmo,
            epsilon,
            callback,
            t,
            primal,
            tot_time,
            f,
            grad!,
            active_set,
        )
        return x, v, phi_value, dual_gap, should_fw_step, should_continue
    end

    # Reformulate convex hull problem, as a quadratic prgroam over the simplex with quadratic term M and linear term b
    M, b = build_reduced_problem(
        active_set.atoms,
        corrective_step.hessian,
        active_set.weights,
        gradient,
        progress_threshold,
    )
    # If the reduced problem is build because the strong-Wolfe gap is too small, proceed to the FW or dual step
    if M === nothing
        v, dual_gap, phi_value, should_fw_step, should_continue = _corrective_fw_or_dual_step(
            corrective_step.lazy,
            corrective_step.lazy_tolerance,
            gradient,
            x,
            v,
            dual_gap,
            phi_value,
            lmo,
            epsilon,
            callback,
            t,
            primal,
            tot_time,
            f,
            grad!,
            active_set,
        )
        return x, v, phi_value, dual_gap, should_fw_step, should_continue
    end

    S = schur(M)
    L_reduced = maximum(real, S.values)
    mu_reduced = max(minimum(real, S.values), zero(L_reduced))
    k = length(active_set.weights)
    λ = copy(active_set.weights)
    x_prev = copy(x)
    gamma = inv(L_reduced)

    use_accelerated = corrective_step.accelerated && L_reduced / mu_reduced > one(L_reduced)
    if use_accelerated
        if corrective_step.y === nothing || length(corrective_step.y) != k
            corrective_step.y = copy(λ)
            corrective_step.alpha = 0.0
        end
        y = corrective_step.y
        grad_y = b + M * y
        λ_new, drop_indices = _simplex_projection_with_drops(y .- grad_y / L_reduced)
        if mu_reduced < 1.0e-3
            alpha_old = corrective_step.alpha
            corrective_step.alpha = 0.5 * (1 + sqrt(1 + 4 * alpha_old^2))
            gamma = (alpha_old - 1.0) / corrective_step.alpha
        else
            q = mu_reduced / L_reduced
            sq = sqrt(q)
            gamma = (1 - sq) / (1 + sq)
        end
        diff = λ_new - λ
        @. y = λ_new + gamma * diff
        λ = λ_new
    else
        corrective_step.y = nothing
        corrective_step.alpha = 0.0
        grad_λ = b + M * λ
        λ, drop_indices = _simplex_projection_with_drops(λ .- grad_λ / L_reduced)
        gamma = inv(L_reduced)
    end

    update_weights!(active_set, λ)
    deleteat!(active_set, drop_indices)
    if corrective_step.y !== nothing && !isempty(drop_indices)
        deleteat!(corrective_step.y, drop_indices)
    end
    compute_active_set_iterate!(active_set)
    x = get_active_set_iterate(active_set)
    d = muladd_memory_mode(memory_mode, d, x_prev, x)
    if callback !== nothing
        state = CallbackState(
            t,
            primal,
            primal - phi_value,
            phi_value,
            tot_time,
            x_prev,
            x,
            d,
            gamma,
            f,
            grad!,
            lmo,
            gradient,
            ST_SIMPLEXDESCENT,
        )
        should_continue = callback(state, active_set)
    end
    if mod(t, renorm_interval) == 0
        active_set_renormalize!(active_set)
        x = compute_active_set_iterate!(active_set)
    end
    return x, v, phi_value, dual_gap, false, should_continue
end

