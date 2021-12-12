
"""
    blended_pairwise_conditional_gradient(f, grad!, lmo, x0; kwargs...)

Implements the BPCG algorithm from [Tsuji, Tanaka, Pokutta](https://arxiv.org/abs/2110.12650).
The method uses an active set of current vertices.
Unlike away-step, it transfers weight from an away vertex to another vertex of the active set.
"""
function blended_pairwise_conditional_gradient(
    f,
    grad!,
    lmo,
    x0;
    line_search::LineSearchMethod=Adaptive(),
    L=Inf,
    gamma0=0,
    step_lim=20,
    epsilon=1e-7,
    max_iteration=10000,
    print_iter=1000,
    trajectory=false,
    verbose=false,
    linesearch_tol=1e-7,
    emphasis::Emphasis=memory,
    gradient=nothing,
    callback=nothing,
    timeout=Inf,
    print_callback=print_callback,
    renorm_interval=1000,
    lazy=false,
    lazy_tolerance=2.0,
)
    # add the first vertex to active set from initialization
    active_set = ActiveSet([(1.0, x0)])

    return blended_pairwise_conditional_gradient(
        f,
        grad!,
        lmo,
        active_set,
        line_search=line_search,
        L=L,
        gamma0=gamma0,
        step_lim=step_lim,
        epsilon=epsilon,
        max_iteration=max_iteration,
        print_iter=print_iter,
        trajectory=trajectory,
        verbose=verbose,
        linesearch_tol=linesearch_tol,
        emphasis=emphasis,
        gradient=gradient,
        callback=callback,
        timeout=timeout,
        print_callback=print_callback,
        renorm_interval=renorm_interval,
        lazy=lazy,
        lazy_tolerance=lazy_tolerance,
    )
end

"""
    blended_pairwise_conditional_gradient(f, grad!, lmo, active_set::ActiveSet; kwargs...)

Warm-starts BPCG with a pre-defined `active_set`.
"""
function blended_pairwise_conditional_gradient(
    f,
    grad!,
    lmo,
    active_set::ActiveSet;
    line_search::LineSearchMethod=Adaptive(),
    L=Inf,
    gamma0=0,
    step_lim=20,
    epsilon=1e-7,
    max_iteration=10000,
    print_iter=1000,
    trajectory=false,
    verbose=false,
    linesearch_tol=1e-7,
    emphasis::Emphasis=memory,
    gradient=nothing,
    callback=nothing,
    timeout=Inf,
    print_callback=print_callback,
    renorm_interval=1000,
    lazy=false,
    lazy_tolerance=2.0,
)

    # format string for output of the algorithm
    format_string = "%6s %13s %14e %14e %14e %14e %14e %14i\n"

    t = 0
    primal = Inf
    x = get_active_set_iterate(active_set)
    tt = regular
    traj_data = []
    if trajectory && callback === nothing
        callback = trajectory_callback(traj_data)
    end
    time_start = time_ns()

    d = similar(x)

    if line_search isa Shortstep && L == Inf
        println("WARNING: Lipschitz constant not set. Prepare to blow up spectacularly.")
    end

    if line_search isa FixedStep && gamma0 == 0
        println("WARNING: gamma0 not set. We are not going to move a single bit.")
    end

    if verbose
        println("\nBlended Pairwise Conditional Gradient Algorithm.")
        num_type = eltype(x)
        println(
            "EMPHASIS: $emphasis STEPSIZE: $line_search EPSILON: $epsilon MAXITERATION: $max_iteration TYPE: $num_type",
        )
        grad_type = typeof(gradient)
        println(
            "GRADIENTTYPE: $grad_type LAZY: $lazy lazy_tolerance: $lazy_tolerance",
        )
        if emphasis == memory
            println("WARNING: In memory emphasis mode iterates are written back into x0!")
        end
        headers =
            ("Type", "Iteration", "Primal", "Dual", "Dual Gap", "Time", "It/sec", "#ActiveSet")
        print_callback(headers, format_string, print_header=true)
    end

    # likely not needed anymore as now the iterates are provided directly via the active set
    if gradient === nothing
        gradient = similar(x)
    end

    grad!(gradient, x)
    v = compute_extreme_point(lmo, gradient)
    # if !lazy, phi is maintained as the global dual gap
    phi = max(0, fast_dot(x, gradient) - fast_dot(v, gradient))
    local_gap = zero(phi)
    gamma = 1.0

    while t <= max_iteration && phi >= max(epsilon, eps())

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

        # compute current iterate from active set
        x = get_active_set_iterate(active_set)
        grad!(gradient, x)

        _, v_local, v_local_loc, _, a_lambda, a, a_loc, _ ,_ =
            active_set_argminmax(active_set, gradient)
        
        local_gap = fast_dot(gradient, a) - fast_dot(gradient, v_local)
        
        if !lazy
            v = compute_extreme_point(lmo, gradient)
            dual_gap = fast_dot(gradient, x) - fast_dot(gradient, v)
            phi = dual_gap
        end
        # minor modification from original paper for improved sparsity
        # (proof follows with minor modification when estimating the step)
        if local_gap ≥ phi / lazy_tolerance
            @. d = a - v_local
            vertex_taken = v_local
            gamma_max = a_lambda
            gamma, L = line_search_wrapper(
                line_search,
                t,
                f,
                grad!,
                x,
                d,
                gradient,
                phi,
                L,
                gamma0,
                linesearch_tol,
                step_lim,
                gamma_max,
            )
            # reached maximum of lambda -> dropping away vertex
            if gamma ≈ gamma_max
                tt = drop
                active_set.weights[v_local_loc] += gamma
                deleteat!(active_set, a_loc)
            else # transfer weight from away to local FW
                tt = pairwise
                active_set.weights[a_loc] -= gamma
                active_set.weights[v_local_loc] += gamma
                @assert active_set_validate(active_set)
            end
            active_set_update_iterate_pairwise!(active_set, gamma, v_local, a)
        else # add to active set
            if lazy # otherwise, v computed above already
                v = compute_extreme_point(lmo, gradient)
            end
            vertex_taken = v
            dual_gap = fast_dot(gradient, x) - fast_dot(gradient, v)
            if (!lazy || dual_gap ≥ phi / lazy_tolerance)
                tt = regular
                @. d = x - v
                gamma_max = one(eltype(x))
                gamma, L = line_search_wrapper(
                    line_search,
                    t,
                    f,
                    grad!,
                    x,
                    d,
                    gradient,
                    dual_gap,
                    L,
                    gamma0,
                    linesearch_tol,
                    step_lim,
                    gamma_max,
                )
                # dropping active set and restarting from singleton
                if gamma ≈ 1.0
                    active_set_initialize!(active_set, v)
                else
                    renorm = mod(t, renorm_interval) == 0
                    active_set_update!(active_set, gamma, v, renorm, nothing)
                end
            else # dual step
                tt = dualstep
                # set to computed dual_gap for consistency between the lazy and non-lazy run.
                # that is ok as we scale with the K = 2.0 default anyways
                phi = dual_gap
            end
        end
        if (
            ((mod(t, print_iter) == 0 || tt == dualstep) == 0 && verbose) ||
            callback !== nothing ||
            !(line_search isa Agnostic || line_search isa Nonconvex || line_search isa FixedStep)
        )
            primal = f(x)
        end
        if callback !== nothing
            state = (
                t=t,
                primal=primal,
                dual=primal - dual_gap,
                dual_gap=phi,
                time=tot_time,
                x=x,
                v=vertex_taken,
                active_set_length=length(active_set),
                gamma=gamma,
            )
            callback(state)
        end

        if verbose && (mod(t, print_iter) == 0 || tt == dualstep)
            if t == 0
                tt = initial
            end
            rep = (
                st[Symbol(tt)],
                string(t),
                Float64(primal),
                Float64(primal - dual_gap),
                Float64(dual_gap),
                tot_time,
                t / tot_time,
                length(active_set),
            )
            print_callback(rep, format_string)
            flush(stdout)
        end
        t += 1
    end

    # recompute everything once more for final verfication / do not record to trajectory though for now!
    # this is important as some variants do not recompute f(x) and the dual_gap regularly but only when reporting
    # hence the final computation.
    # do also cleanup of active_set due to many operations on the same set

    if verbose
        x = get_active_set_iterate(active_set)
        grad!(gradient, x)
        v = compute_extreme_point(lmo, gradient)
        primal = f(x)
        phi = fast_dot(x, gradient) - fast_dot(v, gradient)
        tt = last
        rep = (
            st[Symbol(tt)],
            string(t - 1),
            Float64(primal),
            Float64(primal - dual_gap),
            Float64(dual_gap),
            (time_ns() - time_start) / 1.0e9,
            t / ((time_ns() - time_start) / 1.0e9),
            length(active_set),
        )
        print_callback(rep, format_string)
        flush(stdout)
    end
    active_set_renormalize!(active_set)
    active_set_cleanup!(active_set)
    x = get_active_set_iterate(active_set)
    grad!(gradient, x)
    v = compute_extreme_point(lmo, gradient)
    primal = f(x)
    dual_gap = fast_dot(x, gradient) - fast_dot(v, gradient)
    if verbose
        tt = pp
        rep = (
            st[Symbol(tt)],
            string(t - 1),
            Float64(primal),
            Float64(primal - dual_gap),
            Float64(dual_gap),
            (time_ns() - time_start) / 1.0e9,
            t / ((time_ns() - time_start) / 1.0e9),
            length(active_set),
        )
        print_callback(rep, format_string)
        print_callback(nothing, format_string, print_footer=true)
        flush(stdout)
    end

    return x, v, primal, dual_gap, traj_data, active_set
end
