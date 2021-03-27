
# currently just a copy of fw to be adjusted
# support: fw with tracking of decomposition, afw, pfw over "abstract" functions
# decide in the whether we can lazify that version -> likely possible but will require careful checking. 
# keep lazy variant separate but can be based off of afw
function away_frank_wolfe(
    f,
    grad!,
    lmo,
    x0;
    line_search::LineSearchMethod=adaptive,
    L=Inf,
    gamma0=0,
    K=2.0,
    step_lim=20,
    epsilon=1e-7,
    awaySteps=true,
    lazy=false,
    momentum=nothing,
    max_iteration=10000,
    print_iter=1000,
    trajectory=false,
    verbose=false,
    linesearch_tol=1e-7,
    emphasis::Emphasis=memory,
    gradient=nothing,
    renorm_interval=1000,
)
    function print_header(data)
        @printf(
            "\n───────────────────────────────────────────────────────────────────────────────────────────────────────────────\n"
        )
        @printf(
            "%6s %13s %14s %14s %14s %14s %14s %14s\n",
            data[1],
            data[2],
            data[3],
            data[4],
            data[5],
            data[6],
            data[7],
            data[8],
        )
        @printf(
            "───────────────────────────────────────────────────────────────────────────────────────────────────────────────\n"
        )
    end

    function print_footer()
        @printf(
            "───────────────────────────────────────────────────────────────────────────────────────────────────────────────\n\n"
        )
    end

    function print_iter_func(data)
        @printf(
            "%6s %13s %14e %14e %14e %14e %14e %14i\n",
            st[Symbol(data[1])],
            data[2],
            Float64(data[3]),
            Float64(data[4]),
            Float64(data[5]),
            data[6],
            data[7],
            data[8],
        )
    end

    t = 0
    dual_gap = Inf
    primal = Inf
    x = x0
    active_set = ActiveSet([(1.0, x0)]) # add the first vertex to active set from initialization
    tt = regular
    trajData = []
    time_start = time_ns()

    d = similar(x)

    if line_search == shortstep && L == Inf
        println("WARNING: Lipschitz constant not set. Prepare to blow up spectacularly.")
    end

    if line_search == fixed && gamma0 == 0
        println("WARNING: gamma0 not set. We are not going to move a single bit.")
    end

    if verbose
        println("\nAway-step Frank-Wolfe Algorithm.")
        numType = eltype(x0)
        println(
            "EMPHASIS: $emphasis STEPSIZE: $line_search EPSILON: $epsilon MAXITERATION: $max_iteration TYPE: $numType",
        )
        grad_type = typeof(gradient)
        println(
            "GRADIENTTYPE: $grad_type LAZY: $lazy K: $K MOMENTUM: $momentum AWAYSTEPS: $awaySteps",
        )
        if emphasis == memory
            println("WARNING: In memory emphasis mode iterates are written back into x0!")
        end
        headers =
            ("Type", "Iteration", "Primal", "Dual", "Dual Gap", "Time", "It/sec", "#ActiveSet")
        print_header(headers)
    end

    # likely not needed anymore as now the iterates are provided directly via the active set
    if gradient === nothing
        gradient = similar(x0, float(eltype(x0)))
    end
    gtemp = if momentum !== nothing
        similar(gradient)
    else
        nothing
    end

    x = compute_active_set_iterate(active_set)
    grad!(gradient, x)
    v = compute_extreme_point(lmo, gradient)
    phi_value = fast_dot(x, gradient) - fast_dot(v, gradient)

    while t <= max_iteration && dual_gap >= max(epsilon, eps())

        # compute current iterate from active set
        x = compute_active_set_iterate(active_set)
        if isnothing(momentum)
            grad!(gradient, x)
        else
            grad!(gtemp, x)
            @emphasis(emphasis, gradient = (momentum * gradient) + (1 - momentum) * gtemp)
        end

        if awaySteps
            if lazy
                d, vertex, index, gamma_max, phi_value, away_step_taken, fw_step_taken, tt =
                    lazy_afw_step(x, gradient, lmo, active_set, phi_value; K=K)
            else
                d, vertex, index, gamma_max, phi_value, away_step_taken, fw_step_taken, tt =
                    afw_step(x, gradient, lmo, active_set)
            end
        else
            d, vertex, index, gamma_max, phi_value, away_step_taken, fw_step_taken, tt =
                fw_step(x, gradient, lmo)
        end


        if fw_step_taken || away_step_taken
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

            # cleanup and renormalize every x iterations. Only for the fw steps.
            renorm = mod(t, renorm_interval) == 0

            if away_step_taken
                active_set_update!(active_set, -gamma, vertex, true, index)
            else
                active_set_update!(active_set, gamma, vertex, renorm, index)
            end
        end

        if (
            (mod(t, print_iter) == 0 && verbose) ||
            trajectory ||
            !(line_search == agnostic || line_search == nonconvex || line_search == fixed)
        )
            primal = f(x)
            dual_gap = phi_value
        end

        if trajectory
            push!(
                trajData,
                (
                    t,
                    primal,
                    primal - dual_gap,
                    phi_value,
                    (time_ns() - time_start) / 1.0e9,
                    length(active_set),
                ),
            )
        end


        if verbose && (mod(t, print_iter) == 0 || tt == dualstep)
            if t == 0
                tt = initial
            end
            rep = (
                tt,
                string(t),
                primal,
                primal - dual_gap,
                dual_gap,
                (time_ns() - time_start) / 1.0e9,
                t / ((time_ns() - time_start) / 1.0e9),
                length(active_set),
            )
            print_iter_func(rep)
            flush(stdout)
        end
        t = t + 1
    end

    # recompute everything once more for final verfication / do not record to trajectory though for now! 
    # this is important as some variants do not recompute f(x) and the dual_gap regularly but only when reporting
    # hence the final computation.
    # do also cleanup of active_set due to many operations on the same set

    if verbose
        x = compute_active_set_iterate(active_set)
        grad!(gradient, x)
        v = compute_extreme_point(lmo, gradient)
        primal = f(x)
        dual_gap = fast_dot(x, gradient) - fast_dot(v, gradient)
        tt = last
        rep = (
            tt,
            string(t - 1),
            primal,
            primal - dual_gap,
            dual_gap,
            (time_ns() - time_start) / 1.0e9,
            t / ((time_ns() - time_start) / 1.0e9),
            length(active_set),
        )
        print_iter_func(rep)
        flush(stdout)
    end

    active_set_renormalize!(active_set)
    active_set_cleanup!(active_set)
    x = compute_active_set_iterate(active_set)
    grad!(gradient, x)
    v = compute_extreme_point(lmo, gradient)
    primal = f(x)
    dual_gap = fast_dot(x, gradient) - fast_dot(v, gradient)
    if verbose
        tt = pp
        rep = (
            tt,
            string(t - 1),
            primal,
            primal - dual_gap,
            dual_gap,
            (time_ns() - time_start) / 1.0e9,
            t / ((time_ns() - time_start) / 1.0e9),
            length(active_set),
        )
        print_iter_func(rep)
        print_footer()
        flush(stdout)
    end

    return x, v, primal, dual_gap, trajData, active_set
end

function lazy_afw_step(x, gradient, lmo, active_set, phi; K=2.0)
    v_lambda, v, v_loc, a_lambda, a, a_loc = active_set_argminmax(active_set, gradient)
    #Do lazy FW step
    grad_dot_lazy_fw_vertex = fast_dot(v, gradient)
    grad_dot_x = fast_dot(x, gradient)
    grad_dot_a = fast_dot(a, gradient)
    if grad_dot_x - grad_dot_lazy_fw_vertex >= grad_dot_a - grad_dot_x &&
       grad_dot_x - grad_dot_lazy_fw_vertex >= phi / K
        tt = lazy
        gamma_max = 1
        d = x - v
        vertex = v
        away_step_taken = false
        fw_step_taken = true
        index = v_loc
    else
        #Do away step, as it promises enough progress.
        if grad_dot_a - grad_dot_x > grad_dot_x - grad_dot_lazy_fw_vertex &&
           grad_dot_a - grad_dot_x >= phi / K
            tt = away
            gamma_max = a_lambda / (1 - a_lambda)
            d = a - x
            vertex = a
            away_step_taken = true
            fw_step_taken = false
            index = a_loc
            #Resort to calling the LMO
        else
            v = compute_extreme_point(lmo, gradient)
            # Real dual gap promises enough progress.
            grad_dot_fw_vertex = fast_dot(v, gradient)
            dual_gap = grad_dot_x - grad_dot_fw_vertex
            if dual_gap >= phi / K
                tt = regular
                gamma_max = 1
                d = x - v
                vertex = v
                away_step_taken = false
                fw_step_taken = true
                index = nothing
                #Lower our expectation for progress.
            else
                tt = dualstep
                phi = min(dual_gap, phi / 2.0)
                gamma_max = 0.0
                d = zeros(length(x))
                vertex = v
                away_step_taken = false
                fw_step_taken = false
                index = nothing
            end
        end
    end
    return d, vertex, index, gamma_max, phi, away_step_taken, fw_step_taken, tt
end

function afw_step(x, gradient, lmo, active_set)
    local_v_lambda, local_v, local_v_loc, a_lambda, a, a_loc =
        active_set_argminmax(active_set, gradient)
    away_gap = fast_dot(a, gradient) - fast_dot(x, gradient)
    v = compute_extreme_point(lmo, gradient)
    grad_dot_x = fast_dot(x, gradient)
    away_gap = fast_dot(a, gradient) - grad_dot_x
    dual_gap = grad_dot_x - fast_dot(v, gradient)
    if dual_gap >= away_gap
        tt = regular
        gamma_max = 1
        d = x - v
        vertex = v
        away_step_taken = false
        fw_step_taken = true
        index = nothing
    else
        tt = away
        gamma_max = a_lambda / (1 - a_lambda)
        d = a - x
        vertex = a
        away_step_taken = true
        fw_step_taken = false
        index = a_loc
    end
    return d, vertex, index, gamma_max, dual_gap, away_step_taken, fw_step_taken, tt
end

function fw_step(x, gradient, lmo)
    vertex = compute_extreme_point(lmo, gradient)
    return x - vertex,
    vertex,
    nothing,
    1,
    fast_dot(x, gradient) - fast_dot(vertex, gradient),
    false,
    true,
    regular
end
