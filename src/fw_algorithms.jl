
"""
    frank_wolfe(f, grad!, lmo, x0; ...)

Simplest form of the Frank-Wolfe algorithm.
Returns a tuple `(x, v, primal, dual_gap, traj_data)` with:
- `x` final iterate
- `v` last vertex from the LMO
- `primal` primal value `f(x)`
- `dual_gap` final Frank-Wolfe gap
- `traj_data` vector of trajectory information.
"""
function frank_wolfe(
    f,
    grad!,
    lmo,
    x0;
    line_search::LineSearchMethod=Adaptive(),
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
    emphasis::Emphasis=memory,
    gradient=nothing,
    callback=nothing,
    timeout=Inf,
    print_callback=FrankWolfe.print_callback,
)

    # format string for output of the algorithm
    format_string = "%6s %13s %14e %14e %14e %14e %14e\n"
    t = 0
    dual_gap = Inf
    primal = Inf
    v = []
    x = x0
    tt = regular
    traj_data = []
    if trajectory && callback === nothing
        callback = trajectory_callback(traj_data)
    end
    time_start = time_ns()

    if line_search isa Shortstep && !isfinite(L)
        println("FATAL: Lipschitz constant not set. Prepare to blow up spectacularly.")
    end

    if line_search isa FixedStep && gamma0 == 0
        println("FATAL: gamma0 not set. We are not going to move a single bit.")
    end

    if (!isnothing(momentum) && line_search isa Union{Shortstep,Adaptive,RationalShortstep})
        println(
            "WARNING: Momentum-averaged gradients should usually be used with agnostic stepsize rules.",
        )
    end

    if verbose
        println("\nVanilla Frank-Wolfe Algorithm.")
        numType = eltype(x0)
        println(
            "EMPHASIS: $emphasis STEPSIZE: $line_search EPSILON: $epsilon MAXITERATION: $max_iteration TYPE: $numType",
        )
        grad_type = typeof(gradient)
        println("MOMENTUM: $momentum GRADIENTTYPE: $grad_type")
        if emphasis === memory
            println("WARNING: In memory emphasis mode iterates are written back into x0!")
        end
        headers = ["Type", "Iteration", "Primal", "Dual", "Dual Gap", "Time", "It/sec"]
        print_callback(headers, format_string, print_header=true)
    end

    if emphasis == memory && !isa(x, Union{Array,SparseArrays.AbstractSparseArray})
        # if integer, convert element type to most appropriate float
        if eltype(x) <: Integer
            x = convert(Array{float(eltype(x))}, x)
        else
            x = convert(Array{eltype(x)}, x)
        end
    end
    first_iter = true
    # instanciating container for gradient
    if gradient === nothing
        gradient = similar(x)
    end

    # container for direction
    d = similar(x)
    gtemp = if momentum === nothing
        nothing
    else
        similar(x)
    end
    while t <= max_iteration && dual_gap >= max(epsilon, eps())

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


        if momentum === nothing || first_iter
            grad!(gradient, x)
            if momentum !== nothing
                gtemp .= gradient
            end
        else
            grad!(gtemp, x)
            @emphasis(emphasis, gradient = (momentum * gradient) + (1 - momentum) * gtemp)
        end
        first_iter = false

        v = compute_extreme_point(lmo, gradient)
        # go easy on the memory - only compute if really needed
        if (
            (mod(t, print_iter) == 0 && verbose) ||
            callback !== nothing ||
            line_search isa Shortstep
        )
            primal = f(x)
            dual_gap = fast_dot(x, gradient) - fast_dot(v, gradient)
        end

        if callback !== nothing
            state = (
                t=t,
                primal=primal,
                dual=primal - dual_gap,
                dual_gap=dual_gap,
                time=tot_time,
                x=x,
                v=v,
            )
            callback(state)
        end
        @emphasis(emphasis, d = x - v)

        gamma, L = line_search_wrapper(
            line_search,
            t,
            f,
            grad!,
            x,
            d,
            momentum === nothing ? gradient : gtemp, # use appropriate storage
            dual_gap,
            L,
            gamma0,
            linesearch_tol,
            step_lim,
            one(eltype(x)),
        )

        @emphasis(emphasis, x = x - gamma * d)

        if (mod(t, print_iter) == 0 && verbose)
            tt = regular
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
            )
            print_callback(rep, format_string)

            flush(stdout)
        end
        t = t + 1
    end
    # recompute everything once for final verfication / do not record to trajectory though for now!
    # this is important as some variants do not recompute f(x) and the dual_gap regularly but only when reporting
    # hence the final computation.

    grad!(gradient, x)
    v = compute_extreme_point(lmo, gradient)
    primal = f(x)
    dual_gap = fast_dot(x, gradient) - fast_dot(v, gradient)
    if verbose
        tt = last
        tot_time = (time_ns() - time_start) / 1.0e9
        rep = (
            st[Symbol(tt)],
            string(t - 1),
            Float64(primal),
            Float64(primal - dual_gap),
            Float64(dual_gap),
            tot_time,
            t / tot_time,
        )
        print_callback(rep, format_string)
        print_callback(nothing, format_string, print_footer=true)
        flush(stdout)
    end
    return x, v, primal, dual_gap, traj_data
end


"""
    lazified_conditional_gradient

Similar to [`frank_wolfe`](@ref) but lazyfying the LMO:
each call is stored in a cache, which is looked up first for a good-enough direction.
The cache used is a [`FrankWolfe.MultiCacheLMO`](@ref) or a [`FrankWolfe.VectorCacheLMO`](@ref)
depending on whether the provided `cache_size` option is finite.
"""
function lazified_conditional_gradient(
    f,
    grad!,
    lmo_base,
    x0;
    line_search::LineSearchMethod=Adaptive(),
    L=Inf,
    gamma0=0,
    K=2.0,
    cache_size=Inf,
    greedy_lazy=false,
    epsilon=1e-7,
    max_iteration=10000,
    print_iter=1000,
    trajectory=false,
    verbose=false,
    linesearch_tol=1e-7,
    step_lim=20,
    emphasis::Emphasis=memory,
    gradient=nothing,
    VType=typeof(x0),
    callback=nothing,
    timeout=Inf,
    print_callback=FrankWolfe.print_callback,
)

    # format string for output of the algorithm
    format_string = "%6s %13s %14e %14e %14e %14e %14e %14i\n"

    if isfinite(cache_size)
        lmo = MultiCacheLMO{cache_size,typeof(lmo_base),VType}(lmo_base)
    else
        lmo = VectorCacheLMO{typeof(lmo_base),VType}(lmo_base)
    end

    t = 0
    dual_gap = Inf
    primal = Inf
    v = []
    x = x0
    phi = Inf
    traj_data = []
    if trajectory && callback === nothing
        callback = trajectory_callback(traj_data)
    end
    tt = regular
    time_start = time_ns()

    if line_search isa Shortstep && !isfinite(L)
        println("FATAL: Lipschitz constant not set. Prepare to blow up spectacularly.")
    end

    if line_search isa Agnostic || line_search isa Nonconvex
        println("FATAL: Lazification is not known to converge with open-loop step size strategies.")
    end

    if verbose
        println("\nLazified Conditional Gradients (Frank-Wolfe + Lazification).")
        numType = eltype(x0)
        println(
            "EMPHASIS: $emphasis STEPSIZE: $line_search EPSILON: $epsilon MAXITERATION: $max_iteration K: $K TYPE: $numType",
        )
        grad_type = typeof(gradient)
        println("GRADIENTTYPE: $grad_type CACHESIZE $cache_size GREEDYCACHE: $greedy_lazy")
        if emphasis == memory
            println("WARNING: In memory emphasis mode iterates are written back into x0!")
        end
        headers =
            ["Type", "Iteration", "Primal", "Dual", "Dual Gap", "Time", "It/sec", "Cache Size"]
        print_callback(headers, format_string, print_header=true)
    end

    if emphasis == memory && !isa(x, Union{Array,SparseArrays.AbstractSparseArray})
        x = convert(Array{float(eltype(x))}, x)
    end

    if gradient === nothing
        gradient = similar(x)
    end

    # container for direction
    d = similar(x)

    while t <= max_iteration && dual_gap >= max(epsilon, eps())

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

        grad!(gradient, x)

        threshold = fast_dot(x, gradient) - phi / K

        # go easy on the memory - only compute if really needed
        if ((mod(t, print_iter) == 0 && verbose ) || callback !== nothing)
            primal = f(x)
        end

        v = compute_extreme_point(lmo, gradient, threshold=threshold, greedy=greedy_lazy)
        tt = lazy
        if fast_dot(v, gradient) > threshold
            tt = dualstep
            dual_gap = fast_dot(x, gradient) - fast_dot(v, gradient)
            phi = min(dual_gap, phi / 2)
        end

        if callback !== nothing
            state = (
                t=t,
                primal=primal,
                dual=primal - dual_gap,
                dual_gap=dual_gap,
                time=tot_time,
                cache_size=length(lmo),
                x=x,
                v=v,
            )
            callback(state)
        end

        @emphasis(emphasis, d = x - v)

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
            1.0,
        )

        @emphasis(emphasis, x = x - gamma * d)

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
                length(lmo),
            )
            print_callback(rep, format_string)
            flush(stdout)
        end
        t += 1
    end

    # recompute everything once for final verfication / do not record to trajectory though for now!
    # this is important as some variants do not recompute f(x) and the dual_gap regularly but only when reporting
    # hence the final computation.
    grad!(gradient, x)
    v = compute_extreme_point(lmo, gradient)
    primal = f(x)
    dual_gap = fast_dot(x, gradient) - fast_dot(v, gradient)

    if verbose 
        tt = last
        tot_time = (time_ns() - time_start) / 1.0e9
        rep = (
            st[Symbol(tt)],
            string(t - 1),
            Float64(primal),
            Float64(primal - dual_gap),
            Float64(dual_gap),
            tot_time,
            t / tot_time,
            length(lmo),
        )
        print_callback(rep, format_string)
        if verbose
            print_callback(nothing, format_string, print_footer=true)
        end
        flush(stdout)
    end
    return x, v, primal, dual_gap, traj_data
end

"""
    stochastic_frank_wolfe(f::StochasticObjective, lmo, x0; ...)

Stochastic version of Frank-Wolfe, evaluates the objective and gradient stochastically,
implemented through the [FrankWolfe.StochasticObjective](@ref) interface.
"""
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
    callback=nothing,
    timeout=Inf,
    print_callback=FrankWolfe.print_callback,
)

    # format string for output of the algorithm
    format_string = "%6s %13s %14e %14e %14e %14e %14e\n"

    t = 0
    dual_gap = Inf
    primal = Inf
    v = []
    x = x0
    tt = regular
    traj_data = []
    if trajectory && callback === nothing
        callback = trajectory_callback(traj_data)
    end
    dx = similar(x0) # Array{eltype(x0)}(undef, length(x0))
    time_start = time_ns()

    if line_search == Shortstep && L == Inf
        println("FATAL: Lipschitz constant not set. Prepare to blow up spectacularly.")
    end

    if line_search == FixedStep && gamma0 == 0
        println("FATAL: gamma0 not set. We are not going to move a single bit.")
    end

    if verbose
        println("\nStochastic Frank-Wolfe Algorithm.")
        numType = eltype(x0)
        println(
            "EMPHASIS: $emphasis STEPSIZE: $line_search EPSILON: $epsilon max_iteration: $max_iteration TYPE: $numType",
        )
        # TODO: needs to fix
        grad_type = typeof(nothing)
        println("GRADIENTTYPE: $grad_type MOMENTUM: $momentum BATCHSIZE: $batch_size ")
        if emphasis == memory
            println("WARNING: In memory emphasis mode iterates are written back into x0!")
        end
        headers = ("Type", "Iteration", "Primal", "Dual", "Dual Gap", "Time", "It/sec")
        print_callback(headers, format_string, print_header=true)
    end

    if emphasis == memory && !isa(x, Array)
        x = convert(Array{promote_type(eltype(x), Float64)}, x)
    end
    first_iter = true
    gradient = 0
    while t <= max_iteration && dual_gap >= max(epsilon, eps())

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
           callback !== nothing ||
           !(line_search isa Agnostic || line_search isa Nonconvex || line_search isa FixedStep)
            primal = compute_value(f, x, full_evaluation=true)
            dual_gap = fast_dot(x, gradient) - fast_dot(v, gradient)
        end

        if callback !== nothing
            state = (
                t=t,
                primal=primal,
                dual=primal - dual_gap,
                dual_gap=dual_gap,
                time=tot_time,
                x=x,
                v=v,
            )
            callback(state)
        end

        if line_search isa Agnostic
            gamma = 2 // (2 + t)
        elseif line_search isa Nonconvex
            gamma = 1 / sqrt(t + 1)
        elseif line_search isa Shortstep
            gamma = dual_gap / (L * norm(x - v)^2)
        elseif line_search isa RationalShortstep
            rat_dual_gap = sum((x - v) .* gradient)
            gamma = rat_dual_gap // (L * sum((x - v) .^ 2))
        elseif line_search isa FixedStep
            gamma = gamma0
        end

        @emphasis(emphasis, x = (1 - gamma) * x + gamma * v)

        if mod(t, print_iter) == 0 && verbose
            tt = regular
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
            )
            print_callback(rep, format_string)
            flush(stdout)
        end
        t += 1
    end
    # recompute everything once for final verfication / no additional callback call
    # this is important as some variants do not recompute f(x) and the dual_gap regularly but only when reporting
    # hence the final computation.
    # last computation done with full evaluation for exact gradient

    (primal, gradient) = compute_value_gradient(f, x, full_evaluation=true)
    v = compute_extreme_point(lmo, gradient)
    # @show (gradient, primal)
    dual_gap = fast_dot(x, gradient) - fast_dot(v, gradient)
    if verbose
        tt = last
        tot_time = (time_ns() - time_start) / 1.0e9
        rep = (
            st[Symbol(tt)],
            string(t - 1),
            Float64(primal),
            Float64(primal - dual_gap),
            Float64(dual_gap),
            tot_time,
            t / tot_time,
        )
        print_callback(rep, format_string)
        print_callback(nothing, format_string, print_footer=true)
        flush(stdout)
    end
    return x, v, primal, dual_gap, traj_data
end
