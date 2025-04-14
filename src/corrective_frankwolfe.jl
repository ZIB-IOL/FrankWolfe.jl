
"""
    corrective_frankwolfe(f, grad!, lmo, corrective_step, active_set::AS; kwargs...)

A corrective Frank-Wolfe variant with corrective step defined by `corrective_step`.

A corrective FW algorithm alternates between a standard FW step at which a vertex is added to the active set and a corrective step at which an update is performed in the convex hull of current vertices.
Examples of corrective FW algorithms include blended (pairwise) conditional gradients, away-step Frank-Wolfe, and fully-corrective Frank-Wolfe.
"""
function corrective_frankwolfe(
    f,
    grad!,
    lmo,
    corrective_step::CorrectiveStep,
    active_set::AbstractActiveSet{AT,R};
    line_search::LineSearchMethod=Adaptive(),
    epsilon=1e-7,
    max_iteration=10000,
    print_iter=1000,
    trajectory=false,
    verbose=false,
    memory_mode::MemoryEmphasis=InplaceEmphasis(),
    gradient=nothing,
    callback=nothing,
    traj_data=[],
    timeout=Inf,
    renorm_interval=1000,
    linesearch_workspace=nothing,
    weight_purge_threshold=weight_purge_threshold_default(R),
    extra_vertex_storage=nothing,
    add_dropped_vertices=false,
    use_extra_vertex_storage=false,
    recompute_last_vertex=true,
) where {AT,R}

    # format string for output of the algorithm
    format_string = "%6s %13s %14e %14e %14e %14e %14e %14i\n"
    headers = ("Type", "Iteration", "Primal", "Dual", "Dual Gap", "Time", "It/sec", "#ActiveSet")
    function format_state(state, active_set, args...)
        rep = (
            steptype_string[Symbol(state.step_type)],
            string(state.t),
            Float64(state.primal),
            Float64(state.primal - state.dual_gap),
            Float64(state.dual_gap),
            state.time,
            state.t / state.time,
            length(active_set),
        )
        return rep
    end

    if trajectory
        callback = make_trajectory_callback(callback, traj_data)
    end

    if verbose
        callback = make_print_callback(callback, print_iter, headers, format_string, format_state)
    end

    t = 0
    compute_active_set_iterate!(active_set)
    x = get_active_set_iterate(active_set)
    primal = convert(eltype(x), Inf)
    step_type = ST_REGULAR
    time_start = time_ns()

    d = similar(x)

    if gradient === nothing
        gradient = collect(x)
    end

    if verbose
        println("\nCorrective Frank-Wolfe Algorithm with $(nameof(typeof(corrective_step))) correction.")
        NumType = eltype(x)
        println(
            "MEMORY_MODE: $memory_mode STEPSIZE: $line_search EPSILON: $epsilon MAXITERATION: $max_iteration TYPE: $NumType",
        )
        grad_type = typeof(gradient)
        println("GRADIENTTYPE: $grad_type")
        println("LMO: $(typeof(lmo))")
    end

    grad!(gradient, x)
    v = compute_extreme_point(lmo, gradient)
    # if not a lazy corrector, phi is maintained as the global dual gap
    phi = max(0, fast_dot(x, gradient) - fast_dot(v, gradient))
    dual_gap = phi
    gamma = one(phi)

    if linesearch_workspace === nothing
        linesearch_workspace = build_linesearch_workspace(line_search, x, gradient)
    end

    if extra_vertex_storage === nothing
        use_extra_vertex_storage = add_dropped_vertices = false
    end

    while t <= max_iteration && phi >= max(epsilon, eps(epsilon))

        # managing time limit
        time_at_loop = time_ns()
        if t == 0
            time_start = time_at_loop
        end
        # time is measured at beginning of loop for consistency throughout all algorithms
        tot_time = (time_at_loop - time_start) / 1e9

        if timeout < Inf
            if tot_time ≥ timeout
                if verbose
                    @info "Time limit reached"
                end
                break
            end
        end

        #####################
        t += 1

        # compute current iterate from active set
        x = get_active_set_iterate(active_set)
        primal = f(x)
        if t > 1
            grad!(gradient, x)
        end

        should_compute_vertex = prepare_corrective_step(corrective_step, f, grad!, gradient, active_set, t, lmo, primal, phi)

        if should_compute_vertex && t > 1
            v = compute_extreme_point(lmo, gradient)
            dual_gap = fast_dot(gradient, x) - fast_dot(gradient, v)
            phi = dual_gap
        end
        # use the step defined by the corrective step type
        x, v, phi, dual_gap, should_fw_step, should_continue = run_corrective_step(corrective_step, f, grad!, gradient, x, v, dual_gap, active_set, t, lmo, line_search, linesearch_workspace, primal, phi, tot_time, callback, renorm_interval, memory_mode, epsilon, d)
        # interrupt from callback
        if should_continue === false
            break
        end
        if should_fw_step
            step_type = ST_REGULAR
            # if we are about to exit, compute dual_gap with the cleaned-up x
            if dual_gap ≤ epsilon
                active_set_renormalize!(active_set)
                active_set_cleanup!(active_set; weight_purge_threshold=weight_purge_threshold)
                compute_active_set_iterate!(active_set)
                x = get_active_set_iterate(active_set)
                grad!(gradient, x)
                dual_gap = fast_dot(gradient, x) - fast_dot(gradient, v)
            end
            if dual_gap ≥ epsilon
                d = muladd_memory_mode(memory_mode, d, x, v)

                gamma = perform_line_search(
                    line_search,
                    t,
                    f,
                    grad!,
                    gradient,
                    x,
                    d,
                    one(eltype(x)),
                    linesearch_workspace,
                    memory_mode,
                )
                if gamma ≈ 1
                    step_type = ST_DROP
                end
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
                    if callback(state, active_set) === false
                        break
                    end
                end

                # dropping active set and restarting from singleton
                if gamma ≈ 1.0
                    active_set_initialize!(active_set, v)
                else
                    renorm = mod(t, renorm_interval) == 0
                    active_set_update!(active_set, gamma, v, renorm, nothing)
                    if renorm
                        active_set_renormalize!(active_set)
                        x = compute_active_set_iterate!(active_set)
                    end
                end
            end
        end
    end

    # recompute everything once more for final verfication / do not record to trajectory though for now!
    # this is important as some variants do not recompute f(x) and the dual_gap regularly but only when reporting
    # hence the final computation.
    # do also cleanup of active_set due to many operations on the same set

    if verbose
        compute_active_set_iterate!(active_set)
        x = get_active_set_iterate(active_set)
        grad!(gradient, x)
        v = compute_extreme_point(lmo, gradient)
        primal = f(x)
        phi_new = fast_dot(x, gradient) - fast_dot(v, gradient)
        phi = phi_new < phi ? phi_new : phi
        step_type = ST_LAST
        tot_time = (time_ns() - time_start) / 1e9
        if callback !== nothing
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
            callback(state, active_set)
        end
    end
    active_set_renormalize!(active_set)
    active_set_cleanup!(active_set; weight_purge_threshold=weight_purge_threshold)
    compute_active_set_iterate!(active_set)
    x = get_active_set_iterate(active_set)
    grad!(gradient, x)
    # otherwise values are maintained to last iteration
    if recompute_last_vertex
        v = compute_extreme_point(lmo, gradient)
        primal = f(x)
        dual_gap = fast_dot(x, gradient) - fast_dot(v, gradient)
    end
    step_type = ST_POSTPROCESS
    tot_time = (time_ns() - time_start) / 1e9
    if callback !== nothing
        state = CallbackState(
            t,
            primal,
            primal - dual_gap,
            dual_gap,
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
        callback(state, active_set)
    end

    return (x=x, v=v, primal=primal, dual_gap=dual_gap, traj_data=traj_data, active_set=active_set)
end
