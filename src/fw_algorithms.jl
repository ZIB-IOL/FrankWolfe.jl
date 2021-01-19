
function stochastic_frank_wolfe(
    f::StochasticObjective,
    lmo,
    x0;
    stepSize::LSMethod = agnostic,
    L = Inf,
    gamma0 = 0,
    stepLim = 20,
    momentum = nothing,
    epsilon = 1e-7,
    maxIt = 10000,
    printIt = 1000,
    trajectory = false,
    verbose = false,
    lsTol = 1e-7,
    emph::Emph = blas,
    rng = Random.GLOBAL_RNG,
    batch_size = length(f.xs) ÷ 10 + 1,
    full_evaluation = false,
)
    function headerPrint(data)
        @printf(
            "\n───────────────────────────────────────────────────────────────────────────────────\n"
        )
        @printf(
            "%6s %13s %14s %14s %14s %14s\n",
            data[1],
            data[2],
            data[3],
            data[4],
            data[5],
            data[6]
        )
        @printf(
            "───────────────────────────────────────────────────────────────────────────────────\n"
        )
    end

    function footerPrint()
        @printf(
            "───────────────────────────────────────────────────────────────────────────────────\n\n"
        )
    end

    function itPrint(data)
        @printf(
            "%6s %13s %14e %14e %14e %14e\n",
            st[Symbol(data[1])],
            data[2],
            Float64(data[3]),
            Float64(data[4]),
            Float64(data[5]),
            data[6]
        )
    end

    t = 0
    dualGap = Inf
    primal = Inf
    v = []
    x = x0
    tt = regular
    trajData = []
    dx = similar(x0) # Array{eltype(x0)}(undef, length(x0))
    timeEl = time_ns()

    if stepSize === shortstep && L == Inf
        println("WARNING: Lipschitz constant not set. Prepare to blow up spectacularly.")
    end

    if stepSize === fixed && gamma0 == 0
        println("WARNING: gamma0 not set. We are not going to move a single bit.")
    end

    if verbose
        println("\nStochastic Frank-Wolfe Algorithm.")
        numType = eltype(x0)
        println(
            "EMPHASIS: $emph STEPSIZE: $stepSize EPSILON: $epsilon MAXIT: $maxIt TYPE: $numType",
        )
        println("BATCHSIZE: $batch_size")
        headers = ["Type", "Iteration", "Primal", "Dual", "Dual Gap", "Time"]
        headerPrint(headers)
    end

    if emph === memory && !isa(x, Array)
        x = convert(Vector{promote_type(eltype(x), Float64)}, x)
    end
    first_iter = true
    gradient = 0
    while t <= maxIt && dualGap >= max(epsilon, eps())

        if momentum === nothing || first_iter
            gradient = compute_gradient(
                f,
                x,
                rng = rng,
                batch_size = batch_size,
                full_evaluation = full_evaluation,
            )
        else
            @emphasis(
                emph,
                gradient =
                    (momentum * gradient) .+
                    (1 - momentum) * compute_gradient(
                        f,
                        x,
                        rng = rng,
                        batch_size = batch_size,
                        full_evaluation = full_evaluation,
                    )
            )
        end
        first_iter = false

        v = compute_extreme_point(lmo, gradient)

        # go easy on the memory - only compute if really needed
        if (mod(t, printIt) == 0 && verbose) ||
           trajectory ||
           !(stepSize == agnostic || stepSize == nonconvex || stepSize == fixed)
            primal = compute_value(f, x, full_evaluation = true)
            dualGap = dot(x, gradient) - dot(v, gradient)
        end

        if trajectory === true
            append!(
                trajData,
                [t, primal, primal - dualGap, dualGap, (time_ns() - timeEl) / 1.0e9],
            )
        end

        if stepSize === agnostic
            gamma = 2 // (2 + t)
        elseif stepSize === goldenratio
            _, gamma = segmentSearch(f, grad, x, v, lsTol = lsTol)
        elseif stepSize === backtracking
            _, gamma = backtrackingLS(f, grad, x, v, lsTol = lsTol, stepLim = stepLim)
        elseif stepSize === nonconvex
            gamma = 1 / sqrt(t + 1)
        elseif stepSize === shortstep
            gamma = dualGap / (L * norm(x - v)^2)
        elseif stepSize === rationalshortstep
            ratDualGap = sum((x - v) .* gradient)
            gamma = ratDualGap // (L * sum((x - v) .^ 2))
        elseif stepSize === fixed
            gamma = gamma0
        end

        @emphasis(emph, x = (1 - gamma) * x + gamma * v)

        if mod(t, printIt) == 0 && verbose
            tt = regular
            if t === 0
                tt = initial
            end
            rep = [
                tt,
                string(t),
                primal,
                primal - dualGap,
                dualGap,
                (time_ns() - timeEl) / 1.0e9,
            ]
            itPrint(rep)
            flush(stdout)
        end
        t = t + 1
    end
    # recompute everything once for final verfication / do not record to trajectory though for now!
    # this is important as some variants do not recompute f(x) and the dualGap regularly but only when reporting
    # hence the final computation.
    # last computation done with full evaluation for exact gradient

    (primal, gradient) = compute_value_gradient(f, x, full_evaluation = true)
    v = compute_extreme_point(lmo, gradient)
    @show (gradient, primal)
    dualGap = dot(x, gradient) - dot(v, gradient)
    if verbose
        tt = last
        rep = [tt, "", primal, primal - dualGap, dualGap, (time_ns() - timeEl) / 1.0e9]
        itPrint(rep)
        footerPrint()
        flush(stdout)
    end
    return x, v, primal, dualGap, trajData
end
