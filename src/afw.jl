
# currently just a copy of fw to be adjusted
# support: fw with tracking of decomposition, afw, pfw over "abstract" functions
# decide in the whether we can lazify that version -> likely possible but will require careful checking. 
# keep lazy variant separate but can be based off of afw

function afw(
    f,
    grad,
    lmo,
    x0;
    line_search::LineSearchMethod=adaptive,
    awaySteps=true,
    L=Inf,
    gamma0=0,
    step_lim=20,
    momentum=nothing,
    epsilon=1e-7,
    max_iteration=10000,
    print_iter=1000,
    trajectory=false,
    verbose=false,
    linesearch_tol=1e-7,
    emphasis::Emphasis=blas,
)
    function print_header(data)
        @printf(
            "\n───────────────────────────────────────────────────────────────────────────────────────────────\n"
        )
        @printf(
            "%6s %13s %14s %14s %14s %14s %14s\n",
            data[1],
            data[2],
            data[3],
            data[4],
            data[5],
            data[6],
            data[7],
        )
        @printf(
            "───────────────────────────────────────────────────────────────────────────────────────────────\n"
        )
    end

    function print_footer()
        @printf(
            "───────────────────────────────────────────────────────────────────────────────────────────────\n\n"
        )
    end

    function print_iter_func(data)
        @printf(
            "%6s %13s %14e %14e %14e %14e %14i\n",
            st[Symbol(data[1])],
            data[2],
            Float64(data[3]),
            Float64(data[4]),
            Float64(data[5]),
            data[6],
            data[7],
        )
    end

    t = 0
    dual_gap = Inf
    primal = Inf
    x = x0
    active_set = ActiveSet([(1.0, x0)]) # add the first vertex to active set from initialization
    tt:StepType = regular
    trajData = []
    time_start = time_ns()

    first_iter = true
    gradient = 0
    d = 0 # working direction
    away_step_taken = false # flag whether the current step is an away step

    if line_search === shortstep && L == Inf
        println("WARNING: Lipschitz constant not set. Prepare to blow up spectacularly.")
    end

    if line_search === fixed && gamma0 == 0
        println("WARNING: gamma0 not set. We are not going to move a single bit.")
    end

    if verbose
        println("\nAway-step Frank-Wolfe Algorithm.")
        numType = eltype(x0)
        println(
            "EMPHASIS: $emphasis STEPSIZE: $line_search EPSILON: $epsilon max_iteration: $max_iteration TYPE: $numType",
        )
        println("MOMENTUM: $momentum AWAYSTEPS: $awaySteps")
        if emphasis === memory
            println("WARNING: In memory emphasis mode iterates are written back into x0!")
        end
        headers = ("Type", "Iteration", "Primal", "Dual", "Dual Gap", "Time", "#ActiveSet")
        print_header(headers)
    end

    # likely not needed anymore as now the iterates are provided directly via the active set
    if emphasis === memory && !isa(x, Array)
        x = convert(Vector{promote_type(eltype(x), Float64)}, x)
    end

    while t <= max_iteration && dual_gap >= max(epsilon, eps())

        # compute current iterate from active set
        x = compute_active_set_iterate(active_set)

        if isnothing(momentum) || first_iter
            gradient = grad(x)
        else
            @emphasis(emphasis, gradient = (momentum * gradient) .+ (1 - momentum) * grad(x))
        end
        first_iter = false

        v = compute_extreme_point(lmo, gradient)

        # go easy on the memory - only compute if really needed
        if (
            (mod(t, print_iter) == 0 && verbose) ||
            awaySteps ||
            trajectory ||
            !(line_search == agnostic || line_search == nonconvex || line_search == fixed)
        )
            primal = f(x)
            dual_gap = dot(x, gradient) - dot(v, gradient)
        end

        if trajectory
            push!(
                trajData,
                (
                    t,
                    primal,
                    primal - dual_gap,
                    dual_gap,
                    (time_ns() - time_start) / 1.0e9,
                    length(active_set),
                ),
            )
        end

        # default is a FW step
        # used for clipping the step
        tt = regular
        gamma_max = 1
        d = x - v
        away_step_taken = false

        # above we have already compute the FW vetex and the dual_gap. now we need to 
        # compute the away vertex and the away gap
        lambda, a, i = active_set_argmin(active_set, -gradient)
        away_gap = dot(a, gradient) - dot(x, gradient)

        # if away_gap is larger than dual_gap and we do awaySteps, then away step promises more progress
        if dual_gap < away_gap && awaySteps
            tt = away
            gamma_max = lambda / (1 - lambda)
            d = a - x
            away_step_taken = true
        end


        if line_search === agnostic
            gamma = 2 // (2 + t)
        elseif line_search === goldenratio
            _, gamma = segmentSearch(f, grad, x, v, linesearch_tol=linesearch_tol)
        elseif line_search === backtracking
            _, gamma = backtrackingLS(f, gradient, x, v, linesearch_tol=linesearch_tol, step_lim=step_lim)
        elseif line_search === nonconvex
            gamma = 1 / sqrt(t + 1)
        elseif line_search === shortstep
            gap = dot(gradient, d)
            gamma = gap / (L * norm(d)^2)
        elseif line_search === rationalshortstep
            ratDualGap = sum(d .* gradient)
            gamma = ratDualGap // (L * sum(d .^ 2))
        elseif line_search === fixed
            gamma = gamma0
        elseif line_search === adaptive
            L, gamma = adaptive_step_size(f, gradient, x, d, L)
        end

        # clipping the step size for the away steps
        gamma = min(gamma_max, gamma)

        if !away_step_taken
            active_set_update!(active_set, gamma, v)
        else
            active_set_update!(active_set, -gamma, a)
        end

        if mod(t, print_iter) == 0 && verbose
            if t === 0
                tt = initial
            end
            rep = (
                tt,
                string(t),
                primal,
                primal - dual_gap,
                dual_gap,
                (time_ns() - time_start) / 1.0e9,
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
        gradient = grad(x)
        v = compute_extreme_point(lmo, gradient)
        primal = f(x)
        dual_gap = dot(x, gradient) - dot(v, gradient)
        tt = last
        rep = (
            tt,
            string(t - 1),
            primal,
            primal - dual_gap,
            dual_gap,
            (time_ns() - time_start) / 1.0e9,
            length(active_set),
        )
        print_iter_func(rep)
        flush(stdout)
    end

    active_set_renormalize!(active_set)
    active_set_cleanup!(active_set)
    x = compute_active_set_iterate(active_set)
    gradient = grad(x)
    v = compute_extreme_point(lmo, gradient)
    primal = f(x)
    dual_gap = dot(x, gradient) - dot(v, gradient)
    if verbose
        tt = pp
        rep = (
            tt,
            string(t - 1),
            primal,
            primal - dual_gap,
            dual_gap,
            (time_ns() - time_start) / 1.0e9,
            length(active_set),
        )
        print_iter_func(rep)
        print_footer()
        flush(stdout)
    end

    return x, v, primal, dual_gap, trajData, active_set
end
