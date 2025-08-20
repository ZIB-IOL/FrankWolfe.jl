
"""
    pairwise_frank_wolfe(f, grad!, lmo, x0; kwargs...)

Frank-Wolfe with pairwise steps.
The algorithm maintains the current iterate as a convex combination of vertices in the
[`FrankWolfe.ActiveSet`](@ref) data structure.
See [M. Besançon, A. Carderera and S. Pokutta 2021](https://arxiv.org/abs/2104.06675) for illustrations of away steps. 
Unlike away-step, it transfers weight from an away vertex to another vertex.

$COMMON_ARGS

$COMMON_KWARGS

$RETURN_ACTIVESET
"""
function pairwise_frank_wolfe(
    f,
    grad!,
    lmo,
    x0;
    line_search::LineSearchMethod=Secant(),
    sparsity_control=2.0,
    epsilon=1e-7,
    lazy=false,
    momentum=nothing,
    max_iteration=10000,
    print_iter=1000,
    trajectory=false,
    verbose=false,
    memory_mode::MemoryEmphasis=InplaceEmphasis(),
    gradient=nothing,
    renorm_interval=1000,
    callback=nothing,
    traj_data=[],
    timeout=Inf,
    weight_purge_threshold=weight_purge_threshold_default(eltype(x0)),
    extra_vertex_storage=nothing,
    add_dropped_vertices=false,
    use_extra_vertex_storage=false,
    linesearch_workspace=nothing,
    recompute_last_vertex=true,
)
    # add the first vertex to active set from initialization
    active_set = ActiveSet([(1.0, x0)])

    # Call the method using an ActiveSet as input
    return pairwise_frank_wolfe(
        f,
        grad!,
        lmo,
        active_set,
        line_search=line_search,
        sparsity_control=sparsity_control,
        epsilon=epsilon,
        lazy=lazy,
        momentum=momentum,
        max_iteration=max_iteration,
        print_iter=print_iter,
        trajectory=trajectory,
        verbose=verbose,
        memory_mode=memory_mode,
        gradient=gradient,
        renorm_interval=renorm_interval,
        callback=callback,
        traj_data=traj_data,
        timeout=timeout,
        weight_purge_threshold=weight_purge_threshold,
        extra_vertex_storage=extra_vertex_storage,
        add_dropped_vertices=add_dropped_vertices,
        use_extra_vertex_storage=use_extra_vertex_storage,
        linesearch_workspace=linesearch_workspace,
        recompute_last_vertex=recompute_last_vertex,
    )
end

# pairwise FrankWolfe with the active set given as parameter
# note: in this case I don't need x0 as it is given by the active set and might otherwise lead to confusion
function pairwise_frank_wolfe(
    f,
    grad!,
    lmo,
    active_set::AbstractActiveSet{AT,R};
    line_search::LineSearchMethod=Secant(),
    sparsity_control=2.0,
    epsilon=1e-7,
    lazy=false,
    momentum=nothing,
    max_iteration=10000,
    print_iter=1000,
    trajectory=false,
    verbose=false,
    memory_mode::MemoryEmphasis=InplaceEmphasis(),
    gradient=nothing,
    renorm_interval=1000,
    callback=nothing,
    traj_data=[],
    timeout=Inf,
    weight_purge_threshold=weight_purge_threshold_default(R),
    extra_vertex_storage=nothing,
    add_dropped_vertices=false,
    use_extra_vertex_storage=false,
    linesearch_workspace=nothing,
    recompute_last_vertex=true,
) where {AT,R}
    # format string for output of the algorithm
    format_string = "%6s %13s %14e %14e %14e %14e %14e %14i\n"
    headers = ("Type", "Iteration", "Primal", "Dual", "Dual Gap", "Time", "It/sec", "#ActiveSet")
    function format_state(state, active_set)
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

    isempty(active_set) && throw(ArgumentError("Empty active set"))
    sparsity_control < 1 && throw(ArgumentError("sparsity_control cannot be smaller than one"))

    if trajectory
        callback = make_trajectory_callback(callback, traj_data)
    end

    if verbose
        callback = make_print_callback(callback, print_iter, headers, format_string, format_state)
    end

    t = 0
    dual_gap = Inf
    primal = Inf
    x = get_active_set_iterate(active_set)
    step_type = ST_REGULAR

    time_start = time_ns()

    d = similar(x)

    if gradient === nothing
        gradient = collect(x)
    end
    gtemp = if momentum !== nothing
        similar(gradient)
    else
        nothing
    end

    if verbose
        println("\nPairwise Frank-Wolfe Algorithm.")
        NumType = eltype(x)
        println(
            "MEMORY_MODE: $memory_mode STEPSIZE: $line_search EPSILON: $epsilon MAXITERATION: $max_iteration TYPE: $NumType",
        )
        grad_type = typeof(gradient)
        println(
            "GRADIENTTYPE: $grad_type LAZY: $lazy sparsity_control: $sparsity_control MOMENTUM: $momentum",
        )
        println("LMO: $(typeof(lmo))")
        if (use_extra_vertex_storage || add_dropped_vertices) && extra_vertex_storage === nothing
            @warn(
                "use_extra_vertex_storage and add_dropped_vertices options are only usable with a extra_vertex_storage storage"
            )
        end
    end

    x = get_active_set_iterate(active_set)
    primal = f(x)
    v = active_set.atoms[1]
    phi_value = convert(eltype(x), Inf)
    gamma = one(phi_value)
    execution_status = STATUS_RUNNING

    if linesearch_workspace === nothing
        linesearch_workspace = build_linesearch_workspace(line_search, x, gradient)
    end
    if extra_vertex_storage === nothing
        use_extra_vertex_storage = add_dropped_vertices = false
    end

    while t <= max_iteration && phi_value >= max(eps(float(typeof(phi_value))), epsilon)
        #####################
        # managing time and Ctrl-C
        #####################
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
                execution_status = STATUS_TIMEOUT
                break
            end
        end

        #####################
        t += 1

        # compute current iterate from active set
        x = get_active_set_iterate(active_set)
        if momentum === nothing
            grad!(gradient, x)
        else
            grad!(gtemp, x)
            @memory_mode(memory_mode, gradient = (momentum * gradient) + (1 - momentum) * gtemp)
        end


        if lazy
            d, fw_vertex, fw_index, away_vertex, away_index, gamma_max, phi_value, step_type =
                lazy_pfw_step(
                    x,
                    gradient,
                    lmo,
                    active_set,
                    phi_value,
                    epsilon,
                    d;
                    use_extra_vertex_storage=use_extra_vertex_storage,
                    extra_vertex_storage=extra_vertex_storage,
                    sparsity_control=sparsity_control,
                    memory_mode=memory_mode,
                )
        else
            d, fw_vertex, fw_index, away_vertex, away_index, gamma_max, phi_value, step_type =
                pfw_step(x, gradient, lmo, active_set, epsilon, d, memory_mode=memory_mode)
        end
        if fw_index === nothing
            fw_index = find_atom(active_set, fw_vertex)
        end

        if gamma ≈ gamma_max && fw_index === -1
            step_type = ST_DUALSTEP
        end

        gamma = 0.0
        if step_type != ST_DUALSTEP
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

            # pairwise update
            active_set_update_pairwise!(
                active_set,
                gamma,
                gamma_max,
                fw_index,
                away_index,
                fw_vertex,
                away_vertex,
                use_extra_vertex_storage,
                extra_vertex_storage,
            )
        end

        if callback !== nothing
            state = CallbackState(
                t,
                primal,
                primal - dual_gap,
                dual_gap,
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
            if callback(state, active_set) === false
                execution_status = STATUS_INTERRUPTED
                break
            end
        end

        if mod(t, renorm_interval) == 0
            active_set_renormalize!(active_set)
            x = compute_active_set_iterate!(active_set)
        end

        if (
            (mod(t, print_iter) == 0 && verbose) ||
            callback !== nothing ||
            !(line_search isa Agnostic || line_search isa Nonconvex || line_search isa FixedStep)
        )
            primal = f(x)
            dual_gap = phi_value
        end
    end

    if phi_value < max(epsilon, eps(float(typeof(phi_value))))
        execution_status = STATUS_OPTIMAL
    elseif t >= max_iteration
        execution_status = STATUS_MAXITER
    end
    if execution_status === STATUS_RUNNING
        @warn "Status not set"
        execution_status = STATUS_OPTIMAL
    end

    # recompute everything once more for final verfication / do not record to trajectory though for now!
    # this is important as some variants do not recompute f(x) and the dual_gap regularly but only when reporting
    # hence the final computation.
    # do also cleanup of active_set due to many operations on the same set

    x = get_active_set_iterate(active_set)
    grad!(gradient, x)
    v = compute_extreme_point(lmo, gradient)
    primal = f(x)
    dual_gap = dot(gradient, x) - dot(gradient, v)
    dual_gap = min(phi_value, dual_gap)
    step_type = ST_LAST
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

    active_set_renormalize!(active_set)
    active_set_cleanup!(
        active_set;
        weight_purge_threshold=weight_purge_threshold,
        add_dropped_vertices=use_extra_vertex_storage,
        vertex_storage=extra_vertex_storage,
    )
    x = get_active_set_iterate(active_set)
    grad!(gradient, x)
    if recompute_last_vertex
        v = compute_extreme_point(lmo, gradient)
        primal = f(x)
        dual_gap = dot(gradient, x) - dot(gradient, v)
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

    return (
        x=x,
        v=v,
        primal=primal,
        dual_gap=dual_gap,
        status=execution_status,
        traj_data=traj_data,
        active_set=active_set,
    )
end

function lazy_pfw_step(
    x,
    gradient,
    lmo,
    active_set,
    phi_value,
    epsilon,
    d;
    use_extra_vertex_storage=false,
    extra_vertex_storage=nothing,
    sparsity_control=2.0,
    memory_mode::MemoryEmphasis=InplaceEmphasis(),
)
    _, v_local, v_local_loc, _, a_lambda, a_local, a_local_loc, _, _ =
        active_set_argminmax(active_set, gradient)
    # We will always have an away vertex determining the steplength. 
    gamma_max = a_lambda
    away_index = a_local_loc
    fw_index = nothing
    grad_dot_x = dot(gradient, x)
    grad_dot_a_local = dot(gradient, a_local)

    # Do lazy pairwise step
    grad_dot_lazy_fw_vertex = dot(gradient, v_local)

    if grad_dot_a_local - grad_dot_lazy_fw_vertex >= phi_value / sparsity_control &&
       grad_dot_a_local - grad_dot_lazy_fw_vertex >= epsilon
        step_type = ST_LAZY
        v = v_local
        d = muladd_memory_mode(memory_mode, d, a_local, v)
        fw_index = v_local_loc
    else
        # optionally: try vertex storage
        if use_extra_vertex_storage
            lazy_threshold = dot(gradient, a_local) - phi_value / sparsity_control
            (found_better_vertex, new_forward_vertex) =
                storage_find_argmin_vertex(extra_vertex_storage, gradient, lazy_threshold)
            if found_better_vertex
                @debug("Found acceptable lazy vertex in storage")
                v = new_forward_vertex
                step_type = ST_LAZYSTORAGE
            else
                v = compute_extreme_point(lmo, gradient)
                step_type = ST_PAIRWISE
            end
        else
            v = compute_extreme_point(lmo, gradient)
            step_type = ST_PAIRWISE
        end

        # Real dual gap promises enough progress.
        grad_dot_fw_vertex = dot(gradient, v)
        dual_gap = grad_dot_x - grad_dot_fw_vertex
        if dual_gap >= phi_value / sparsity_control
            d = muladd_memory_mode(memory_mode, d, a_local, v)
            #Lower our expectation for progress.
        else
            step_type = ST_DUALSTEP
            phi_value = min(dual_gap, phi_value / 2)
        end
    end
    return d, v, fw_index, a_local, away_index, gamma_max, phi_value, step_type
end


function pfw_step(
    x,
    gradient,
    lmo,
    active_set,
    epsilon,
    d;
    memory_mode::MemoryEmphasis=InplaceEmphasis(),
)
    step_type = ST_PAIRWISE
    _, _, _, _, a_lambda, a_local, a_local_loc = active_set_argminmax(active_set, gradient)
    away_vertex = a_local
    away_index = a_local_loc
    # We will always have a away vertex determining the steplength. 
    gamma_max = a_lambda

    v = compute_extreme_point(lmo, gradient)
    fw_vertex = v
    fw_index = nothing
    grad_dot_x = dot(gradient, x)
    dual_gap = grad_dot_x - dot(gradient, v)
    d = muladd_memory_mode(memory_mode, d, a_local, v)
    return d, fw_vertex, fw_index, away_vertex, away_index, gamma_max, dual_gap, step_type
end
