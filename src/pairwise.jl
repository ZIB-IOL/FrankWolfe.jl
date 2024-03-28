
"""
    pairwise_frank_wolfe(f, grad!, lmo, x0; ...)

Frank-Wolfe with pairwise steps.
The algorithm maintains the current iterate as a convex combination of vertices in the
[`FrankWolfe.ActiveSet`](@ref) data structure.
See [M. Besançon, A. Carderera and S. Pokutta 2021](https://arxiv.org/abs/2104.06675) for illustrations of away steps. 
Unlike away-step, it transfers weight from an away vertex to another vertex.
"""
function pairwise_frank_wolfe(
    f,
    grad!,
    lmo,
    x0;
    line_search::LineSearchMethod=Adaptive(),
    lazy_tolerance=2.0,
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
        lazy_tolerance=lazy_tolerance,
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
    active_set::AbstractActiveSet;
    line_search::LineSearchMethod=Adaptive(),
    lazy_tolerance=2.0,
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
    extra_vertex_storage=nothing,
    add_dropped_vertices=false,
    use_extra_vertex_storage=false,
    linesearch_workspace=nothing,
    recompute_last_vertex=true,
)
    # format string for output of the algorithm
    format_string = "%6s %13s %14e %14e %14e %14e %14e %14i\n"
    headers = ("Type", "Iteration", "Primal", "Dual", "Dual Gap", "Time", "It/sec", "#ActiveSet")
    function format_state(state, active_set)
        rep = (
            st[Symbol(state.tt)],
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


    if isempty(active_set)
        throw(ArgumentError("Empty active set"))
    end

    t = 0
    dual_gap = Inf
    primal = Inf
    x = get_active_set_iterate(active_set)
    tt = regular

    if trajectory
        callback = make_trajectory_callback(callback, traj_data)
    end

    if verbose
        callback = make_print_callback(callback, print_iter, headers, format_string, format_state)
    end

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
            "GRADIENTTYPE: $grad_type LAZY: $lazy lazy_tolerance: $lazy_tolerance MOMENTUM: $momentum",
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
                break
            end
        end

        #####################
        t += 1

        # compute current iterate from active set
        x = get_active_set_iterate(active_set)
        if isnothing(momentum)
            grad!(gradient, x)
        else
            grad!(gtemp, x)
            @memory_mode(memory_mode, gradient = (momentum * gradient) + (1 - momentum) * gtemp)
        end

        
        if lazy
            d, fw_vertex, away_vertex, index, gamma_max, phi_value, tt =
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
                    lazy_tolerance=lazy_tolerance,
                    memory_mode=memory_mode,
                )
        else
            d, fw_vertex, away_vertex, index, gamma_max, phi_value, tt =
                pfw_step(x, gradient, lmo, active_set, epsilon, d, memory_mode=memory_mode)
        end

        gamma = 0.0
        if tt != dualstep
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
            tt = gamma ≈ gamma_max ? drop : tt
            # cleanup and renormalize every x iterations. Only for the fw steps.
            renorm = mod(t, renorm_interval) == 0
            active_set_update!(active_set, -gamma, away_vertex, true, index, add_dropped_vertices=use_extra_vertex_storage, vertex_storage=extra_vertex_storage)
            if add_dropped_vertices && gamma == gamma_max
                for vtx in active_set.atoms
                    if vtx != v
                        push!(extra_vertex_storage, vtx)
                    end
                end
            end
            active_set_update!(active_set, gamma, vertex, renorm, index)
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
                tt,
            )
            if callback(state, active_set) === false
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

    # recompute everything once more for final verfication / do not record to trajectory though for now!
    # this is important as some variants do not recompute f(x) and the dual_gap regularly but only when reporting
    # hence the final computation.
    # do also cleanup of active_set due to many operations on the same set

    x = get_active_set_iterate(active_set)
    grad!(gradient, x)
    v = compute_extreme_point(lmo, gradient)
    primal = f(x)
    dual_gap = fast_dot(x, gradient) - fast_dot(v, gradient)
    tt = last
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
            tt,
        )
        callback(state, active_set)
    end

    active_set_renormalize!(active_set)
    active_set_cleanup!(active_set, add_dropped_vertices=use_extra_vertex_storage, vertex_storage=extra_vertex_storage)
    x = get_active_set_iterate(active_set)
    grad!(gradient, x)
    if recompute_last_vertex
        v = compute_extreme_point(lmo, gradient)
        primal = f(x)
        dual_gap = fast_dot(x, gradient) - fast_dot(v, gradient)
    end
    tt = pp
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
            tt,
        )
        callback(state, active_set)
    end

    return (x=x, v=v, primal=primal, dual_gap=dual_gap, traj_data=traj_data, active_set=active_set)
end

function lazy_pfw_step(x, gradient, lmo, active_set, phi, epsilon, d; use_extra_vertex_storage=false, extra_vertex_storage=nothing, lazy_tolerance=2.0, memory_mode::MemoryEmphasis=InplaceEmphasis())
    _, v_local, v_local_loc, _, a_lambda, a_local, a_local_loc, _, _ = active_set_argminmax(active_set, gradient)
    # We will always have a away vertex determining the steplength. 
    gamma_max = a_lambda

    # Do lazy pairwise step
    grad_dot_lazy_fw_vertex = fast_dot(v_local, gradient)
    grad_dot_x = fast_dot(x, gradient)
    grad_dot_a = fast_dot(a_local, gradient)
    if grad_dot_lazy_fw_vertex - grad_dot_lazy_fw_vertex >= phi / lazy_tolerance && grad_dot_lazy_fw_vertex - grad_dot_lazy_fw_vertex >= epsilon
        tt = lazy
        v  = v_local
        d = muladd_memory_mode(memory_mode, d, a_local, v)
        fw_vertex = v_local
        index = v_loc
    else
        # optionally: try vertex storage
        if use_extra_vertex_storage
            lazy_threshold = fast_dot(gradient, a_local) - phi / lazy_tolerance
            (found_better_vertex, new_forward_vertex) = storage_find_argmin_vertex(extra_vertex_storage, gradient, lazy_threshold)
            if found_better_vertex
                @debug("Found acceptable lazy vertex in storage")
                v = new_forward_vertex
                tt = lazylazy
            else
                v = compute_extreme_point(lmo, gradient)
                tt = pairwise
            end
        else
            v = compute_extreme_point(lmo, gradient)
            tt = pairwise
        end
        fw_vertex = v
        index = -1
        
        # Real pairwise gap promises enough progress.
        grad_dot_fw_vertex = fast_dot(v, gradient)
        pairwise_gap = grad_dot_a - grad_dot_fw_vertex
        if pairwise_gap >= phi / lazy_tolerance
            d = muladd_memory_mode(memory_mode, d, a, v)
            #Lower our expectation for progress.
        else
            tt = dualstep
            phi = min(dual_gap, phi / 2.0)
        end
        end
    end
    return d, vertex, index, gamma_max, phi, away_step_taken, fw_step_taken, tt
end


function pfw_step(x, gradient, lmo, active_set, epsilon, d; memory_mode::MemoryEmphasis=InplaceEmphasis())
    tt = pairwise
    _, _, _, _, a_lambda, a_local, a_local_loc = active_set_argminmax(active_set, gradient)
    away_vertex = a_local
    v = compute_extreme_point(lmo, gradient)
    fw_vertex = v
    grad_dot_x = fast_dot(x, gradient)
    # away_gap = fast_dot(a, gradient) - grad_dot_x
    dual_gap = grad_dot_x - fast_dot(v, gradient)
    
    gamma_max = a_lambda
    d = muladd_memory_mode(memory_mode, d, a_local, v)
    index = a_local_loc 
    return d, fw_vertex, away_vertex, index, gamma_max, dual_gap, tt
end
