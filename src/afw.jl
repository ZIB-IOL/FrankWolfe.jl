
# currently just a copy of fw to be adjusted
# support: fw with tracking of decomposition, afw, pfw over "abstract" functions
# decide in the whether we can lazify that version -> likely possible but will require careful checking. 
# keep lazy variant separate but can be based off of afw

function afw(
    f,
    grad,
    lmo,
    x0;
    step_size::LSMethod=adaptive,
    awaySteps=true,
    L=Inf,
    gamma0=0,
    stepLim=20,
    momentum=nothing,
    epsilon=1e-7,
    maxIt=10000,
    printIt=1000,
    trajectory=false,
    verbose=false,
    lsTol=1e-7,
    emph::Emph=blas,
)
    function headerPrint(data)
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

    function footerPrint()
        @printf(
            "───────────────────────────────────────────────────────────────────────────────────────────────\n\n"
        )
    end

    function itPrint(data)
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
    dualGap = Inf
    primal = Inf
    x = x0
    active_set = ActiveSet([(1.0, x0)]) # add the first vertex to active set from initialization
    tt:StepType = regular
    trajData = []
    timeEl = time_ns()

    first_iter = true
    gradient = 0
    d = 0 # working direction
    away_step_taken = false # flag whether the current step is an away step

    if step_size === shortstep && L == Inf
        println("WARNING: Lipschitz constant not set. Prepare to blow up spectacularly.")
    end

    if step_size === fixed && gamma0 == 0
        println("WARNING: gamma0 not set. We are not going to move a single bit.")
    end

    if verbose
        println("\nActive-set Frank-Wolfe Algorithm.")
        numType = eltype(x0)
        println(
            "EMPHASIS: $emph STEPSIZE: $step_size EPSILON: $epsilon MAXIT: $maxIt TYPE: $numType",
        )
        println("MOMENTUM: $momentum AWAYSTEPS: $awaySteps")
        if emph === memory
            println("WARNING: In memory emphasis mode iterates are written back into x0!")
        end
        headers = ("Type", "Iteration", "Primal", "Dual", "Dual Gap", "Time", "#ActiveSet")
        headerPrint(headers)
    end

    # likely not needed anymore as now the iterates are provided directly via the active set
    if emph === memory && !isa(x, Array)
        x = convert(Vector{promote_type(eltype(x), Float64)}, x)
    end

    while t <= maxIt && dualGap >= max(epsilon, eps())

        # compute current iterate from active set
        x = compute_active_set_iterate(active_set)

        if isnothing(momentum) || first_iter
            gradient = grad(x)
        else
            @emphasis(emph, gradient = (momentum * gradient) .+ (1 - momentum) * grad(x))
        end
        first_iter = false

        v = compute_extreme_point(lmo, gradient)

        # go easy on the memory - only compute if really needed
        if (
            (mod(t, printIt) == 0 && verbose) ||
            awaySteps ||
            trajectory ||
            !(step_size == agnostic || step_size == nonconvex || step_size == fixed)
        )
            primal = f(x)
            dualGap = dot(x, gradient) - dot(v, gradient)
        end

        if trajectory === true
            push!(
                trajData,
                (
                    t,
                    primal,
                    primal - dualGap,
                    dualGap,
                    (time_ns() - timeEl) / 1.0e9,
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

        # above we have already compute the FW vetex and the dualGap. now we need to 
        # compute the away vertex and the away gap
        lambda, a, i = active_set_argmin(active_set, -gradient)
        away_gap = dot(a, gradient) - dot(x, gradient)

        # if away_gap is larger than dualGap and we do awaySteps, then away step promises more progress
        if dualGap < away_gap && awaySteps
            tt = away
            gamma_max = lambda / (1 - lambda)
            d = a - x
            away_step_taken = true
        end


        if step_size === agnostic
            gamma = 2 // (2 + t)
        elseif step_size === goldenratio
            nothing, gamma = segmentSearch(f, grad, x, v, lsTol=lsTol)
        elseif step_size === backtracking
            nothing, gamma = backtrackingLS(f, grad, x, v, lsTol=lsTol, stepLim=stepLim)
        elseif step_size === nonconvex
            gamma = 1 / sqrt(t + 1)
        elseif step_size === shortstep
            gap = dot(gradient, d)
            gamma = gap / (L * norm(d)^2)
        elseif step_size === rationalshortstep
            ratDualGap = sum(d .* gradient)
            gamma = ratDualGap // (L * sum(d .^ 2))
        elseif step_size === fixed
            gamma = gamma0
        elseif step_size === adaptive
            L, gamma = adaptive_step_size(f, gradient, x, d, L)
        end

        # clipping the step size for the away steps
        gamma = min(gamma_max, gamma)

        if !away_step_taken
            active_set_update!(active_set, gamma, v)
        else
            active_set_update!(active_set, -gamma, a)
        end

        if mod(t, printIt) == 0 && verbose
            if t === 0
                tt = initial
            end
            rep = (
                tt,
                string(t),
                primal,
                primal - dualGap,
                dualGap,
                (time_ns() - timeEl) / 1.0e9,
                length(active_set),
            )
            itPrint(rep)
            flush(stdout)
        end
        t = t + 1
    end

    # recompute everything once more for final verfication / do not record to trajectory though for now! 
    # this is important as some variants do not recompute f(x) and the dualGap regularly but only when reporting
    # hence the final computation.
    # do also cleanup of active_set due to many operations on the same set

    if verbose
        x = compute_active_set_iterate(active_set)
        gradient = grad(x)
        v = compute_extreme_point(lmo, gradient)
        primal = f(x)
        dualGap = dot(x, gradient) - dot(v, gradient)
        tt = last
        rep = (
            tt,
            string(t - 1),
            primal,
            primal - dualGap,
            dualGap,
            (time_ns() - timeEl) / 1.0e9,
            length(active_set),
        )
        itPrint(rep)
        flush(stdout)
    end

    active_set_renormalize!(active_set)
    active_set_cleanup!(active_set)
    x = compute_active_set_iterate(active_set)
    gradient = grad(x)
    v = compute_extreme_point(lmo, gradient)
    primal = f(x)
    dualGap = dot(x, gradient) - dot(v, gradient)
    if verbose
        tt = pp
        rep = (
            tt,
            string(t - 1),
            primal,
            primal - dualGap,
            dualGap,
            (time_ns() - timeEl) / 1.0e9,
            length(active_set),
        )
        itPrint(rep)
        footerPrint()
        flush(stdout)
    end

    return x, v, primal, dualGap, trajData, active_set
end
