
function stochastic_frank_wolfe(
    f::StochasticObjective,
    lmo,
    x0;
    line_search::LineSearchMethod=nonconvex,
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
    rng=Random.GLOBAL_RNG,
    batch_size=length(f.xs) ÷ 10 + 1,
    full_evaluation=false,
)
    function print_header(data)
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

    function print_footer()
        @printf(
            "───────────────────────────────────────────────────────────────────────────────────\n\n"
        )
    end

    function print_iter_func(data)
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
    dual_gap = Inf
    primal = Inf
    v = []
    x = x0
    tt = regular
    trajData = []
    dx = similar(x0) # Array{eltype(x0)}(undef, length(x0))
    time_start = time_ns()

    if line_search === shortstep && L == Inf
        println("FATAL: Lipschitz constant not set. Prepare to blow up spectacularly.")
    end

    if line_search === fixed && gamma0 == 0
        println("FATAL: gamma0 not set. We are not going to move a single bit.")
    end

    if verbose
        println("\nStochastic Frank-Wolfe Algorithm.")
        numType = eltype(x0)
        println(
            "EMPHASIS: $emphasis STEPSIZE: $line_search EPSILON: $epsilon max_iteration: $max_iteration TYPE: $numType",
        )
        println("MOMENTUM: $momentum BATCHSIZE: $batch_size ")
        if emphasis === memory
            println("WARNING: In memory emphasis mode iterates are written back into x0!")
        end
        headers = ["Type", "Iteration", "Primal", "Dual", "Dual Gap", "Time"]
        print_header(headers)
    end

    if emphasis === memory && !isa(x, Array)
        x = convert(Vector{promote_type(eltype(x), Float64)}, x)
    end
    first_iter = true
    gradient = 0
    while t <= max_iteration && dual_gap >= max(epsilon, eps())

        if momentum === nothing || first_iter
            gradient = compute_gradient(
                f,
                x,
                rng=rng,
                batch_size=batch_size,
                full_evaluation=full_evaluation,
            )
        else
            @emphasis(
                emphasis,
                gradient =
                    (momentum * gradient) .+
                    (1 - momentum) * compute_gradient(
                        f,
                        x,
                        rng=rng,
                        batch_size=batch_size,
                        full_evaluation=full_evaluation,
                    )
            )
        end
        first_iter = false

        v = compute_extreme_point(lmo, gradient)

        # go easy on the memory - only compute if really needed
        if (mod(t, print_iter) == 0 && verbose) ||
           trajectory ||
           !(line_search == agnostic || line_search == nonconvex || line_search == fixed)
            primal = compute_value(f, x, full_evaluation=true)
            dual_gap = dot(x, gradient) - dot(v, gradient)
        end

        if trajectory === true
            push!(trajData, [t, primal, primal - dual_gap, dual_gap, (time_ns() - time_start) / 1.0e9])
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
            gamma = dual_gap / (L * norm(x - v)^2)
        elseif line_search === rationalshortstep
            ratDualGap = sum((x - v) .* gradient)
            gamma = ratDualGap // (L * sum((x - v) .^ 2))
        elseif line_search === fixed
            gamma = gamma0
        end

        @emphasis(emphasis, x = (1 - gamma) * x + gamma * v)

        if mod(t, print_iter) == 0 && verbose
            tt = regular
            if t === 0
                tt = initial
            end
            rep = [tt, string(t), primal, primal - dual_gap, dual_gap, (time_ns() - time_start) / 1.0e9]
            print_iter_func(rep)
            flush(stdout)
        end
        t = t + 1
    end
    # recompute everything once for final verfication / do not record to trajectory though for now!
    # this is important as some variants do not recompute f(x) and the dual_gap regularly but only when reporting
    # hence the final computation.
    # last computation done with full evaluation for exact gradient

    (primal, gradient) = compute_value_gradient(f, x, full_evaluation=true)
    v = compute_extreme_point(lmo, gradient)
    # @show (gradient, primal)
    dual_gap = dot(x, gradient) - dot(v, gradient)
    if verbose
        tt = last
        rep = [tt, string(t - 1), primal, primal - dual_gap, dual_gap, (time_ns() - time_start) / 1.0e9]
        print_iter_func(rep)
        print_footer()
        flush(stdout)
    end
    return x, v, primal, dual_gap, trajData
end
