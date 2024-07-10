
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
    momentum=nothing,
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
    linesearch_workspace=nothing,
    dual_gap_compute_frequency=1,
)

    # header and format string for output of the algorithm
    headers = ["Type", "Iteration", "Primal", "Dual", "Dual Gap", "Time", "It/sec"]
    format_string = "%6s %13s %14e %14e %14e %14e %14e\n"
    function format_state(state)
        rep = (
            steptype_string[Symbol(state.step_type)],
            string(state.t),
            Float64(state.primal),
            Float64(state.primal - state.dual_gap),
            Float64(state.dual_gap),
            state.time,
            state.t / state.time,
        )
        return rep
    end

    t = 0
    dual_gap = Inf
    primal = Inf
    v = []
    x = x0
    step_type = ST_REGULAR

    if trajectory
        callback = make_trajectory_callback(callback, traj_data)
    end

    if verbose
        callback = make_print_callback(callback, print_iter, headers, format_string, format_state)
    end

    time_start = time_ns()

    if (momentum !== nothing && line_search isa Union{Shortstep,Adaptive,Backtracking})
        @warn("Momentum-averaged gradients should usually be used with agnostic stepsize rules.",)
    end

    if verbose
        println("\nVanilla Frank-Wolfe Algorithm.")
        NumType = eltype(x0)
        println(
            "MEMORY_MODE: $memory_mode STEPSIZE: $line_search EPSILON: $epsilon MAXITERATION: $max_iteration TYPE: $NumType",
        )
        grad_type = typeof(gradient)
        println("MOMENTUM: $momentum GRADIENTTYPE: $grad_type")
        println("LMO: $(typeof(lmo))")
        if memory_mode isa InplaceEmphasis
            @info("In memory_mode memory iterates are written back into x0!")
        end
    end
    if memory_mode isa InplaceEmphasis && !isa(x, Union{Array,SparseArrays.AbstractSparseArray})
        # if integer, convert element type to most appropriate float
        if eltype(x) <: Integer
            x = copyto!(similar(x, float(eltype(x))), x)
        else
            x = copyto!(similar(x), x)
        end
    end

    # instanciating container for gradient
    if gradient === nothing
        gradient = collect(x)
    end

    first_iter = true
    if linesearch_workspace === nothing
        linesearch_workspace = build_linesearch_workspace(line_search, x, gradient)
    end

    # container for direction
    d = similar(x)
    gtemp = momentum === nothing ? d : similar(x)

    while t <= max_iteration && dual_gap >= max(epsilon, eps(float(typeof(dual_gap))))

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
            @memory_mode(memory_mode, gradient = (momentum * gradient) + (1 - momentum) * gtemp)
        end

        v = if first_iter
            compute_extreme_point(lmo, gradient)
        else
            compute_extreme_point(lmo, gradient, v=v)
        end

        first_iter = false
        # go easy on runtime - only compute primal and dual if needed
        compute_iter = (
            (mod(t, print_iter) == 0 && verbose) ||
            callback !== nothing ||
            line_search isa Shortstep
        )
        if compute_iter
            primal = f(x)
        end
        if t %  dual_gap_compute_frequency == 0 || compute_iter
            dual_gap = fast_dot(x, gradient) - fast_dot(v, gradient)
        end
        d = muladd_memory_mode(memory_mode, d, x, v)

        gamma = perform_line_search(
            line_search,
            t,
            f,
            grad!,
            gradient,
            x,
            d,
            1.0,
            linesearch_workspace,
            memory_mode,
        )
        t = t + 1
        if callback !== nothing
            state = CallbackState(
                t,
                primal,
                primal - dual_gap,
                dual_gap,
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
            if callback(state) === false
                break
            end
        end

        x = muladd_memory_mode(memory_mode, x, gamma, d)
    end
    # recompute everything once for final verfication / do not record to trajectory though for now!
    # this is important as some variants do not recompute f(x) and the dual_gap regularly but only when reporting
    # hence the final computation.
    step_type = ST_LAST
    grad!(gradient, x)
    v = compute_extreme_point(lmo, gradient, v=v)
    primal = f(x)
    dual_gap = fast_dot(x, gradient) - fast_dot(v, gradient)
    tot_time = (time_ns() - time_start) / 1.0e9
    gamma = perform_line_search(
        line_search,
        t,
        f,
        grad!,
        gradient,
        x,
        d,
        1.0,
        linesearch_workspace,
        memory_mode,
    )
    if callback !== nothing
        state = CallbackState(
            t,
            primal,
            primal - dual_gap,
            dual_gap,
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
        callback(state)
    end

    return (x=x, v=v, primal=primal, dual_gap=dual_gap, traj_data=traj_data)
end


"""
    lazified_conditional_gradient(f, grad!, lmo_base, x0; ...)

Similar to [`FrankWolfe.frank_wolfe`](@ref) but lazyfying the LMO:
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
    lazy_tolerance=2.0,
    cache_size=Inf,
    greedy_lazy=false,
    epsilon=1e-7,
    max_iteration=10000,
    print_iter=1000,
    trajectory=false,
    verbose=false,
    memory_mode::MemoryEmphasis=InplaceEmphasis(),
    gradient=nothing,
    callback=nothing,
    traj_data=[],
    VType=typeof(x0),
    timeout=Inf,
    linesearch_workspace=nothing,
)

    # format string for output of the algorithm
    format_string = "%6s %13s %14e %14e %14e %14e %14e %14i\n"
    headers = ["Type", "Iteration", "Primal", "Dual", "Dual Gap", "Time", "It/sec", "Cache Size"]
    function format_state(state, args...)
        rep = (
            steptype_string[Symbol(state.step_type)],
            string(state.t),
            Float64(state.primal),
            Float64(state.primal - state.dual_gap),
            Float64(state.dual_gap),
            state.time,
            state.t / state.time,
            length(state.lmo),
        )
        return rep
    end

    lmo = VectorCacheLMO{typeof(lmo_base),VType}(lmo_base)
    if isfinite(cache_size)
        Base.sizehint!(lmo.vertices, cache_size)
    end

    t = 0
    dual_gap = Inf
    primal = Inf
    v = []
    x = x0
    phi = Inf
    step_type = ST_REGULAR

    if trajectory
        callback = make_trajectory_callback(callback, traj_data)
    end

    if verbose
        callback = make_print_callback(callback, print_iter, headers, format_string, format_state)
    end
    time_start = time_ns()

    if line_search isa Agnostic || line_search isa Nonconvex
        @warn("Lazification is not known to converge with open-loop step size strategies.")
    end

    if gradient === nothing
        gradient = collect(x)
    end

    if verbose
        println("\nLazified Conditional Gradient (Frank-Wolfe + Lazification).")
        NumType = eltype(x0)
        println(
            "MEMORY_MODE: $memory_mode STEPSIZE: $line_search EPSILON: $epsilon MAXITERATION: $max_iteration lazy_tolerance: $lazy_tolerance TYPE: $NumType",
        )
        grad_type = typeof(gradient)
        println("GRADIENTTYPE: $grad_type CACHESIZE $cache_size GREEDYCACHE: $greedy_lazy")
        println("LMO: $(typeof(lmo))")
        if memory_mode isa InplaceEmphasis
            @info("In memory_mode memory iterates are written back into x0!")
        end
    end

    if memory_mode isa InplaceEmphasis && !isa(x, Union{Array,SparseArrays.AbstractSparseArray})
        if eltype(x) <: Integer
            x = copyto!(similar(x, float(eltype(x))), x)
        else
            x = copyto!(similar(x), x)
        end
    end

    # container for direction
    d = similar(x)
    if linesearch_workspace === nothing
        linesearch_workspace = build_linesearch_workspace(line_search, x, gradient)
    end

    while t <= max_iteration && dual_gap >= max(epsilon, eps(float(eltype(x))))

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

        threshold = fast_dot(x, gradient) - phi / lazy_tolerance

        # go easy on the memory - only compute if really needed
        if ((mod(t, print_iter) == 0 && verbose) || callback !== nothing)
            primal = f(x)
        end

        v = compute_extreme_point(lmo, gradient, threshold=threshold, greedy=greedy_lazy)
        step_type = ST_LAZY
        if fast_dot(v, gradient) > threshold
            step_type = ST_DUALSTEP
            dual_gap = fast_dot(x, gradient) - fast_dot(v, gradient)
            phi = min(dual_gap, phi / 2)
        end

        d = muladd_memory_mode(memory_mode, d, x, v)

        gamma = perform_line_search(
            line_search,
            t,
            f,
            grad!,
            gradient,
            x,
            d,
            1.0,
            linesearch_workspace,
            memory_mode,
        )

        t += 1
        if callback !== nothing
            state = CallbackState(
                t,
                primal,
                primal - dual_gap,
                dual_gap,
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
            if callback(state) === false
                break
            end
        end

        x = muladd_memory_mode(memory_mode, x, gamma, d)
    end

    # recompute everything once for final verfication / do not record to trajectory though for now!
    # this is important as some variants do not recompute f(x) and the dual_gap regularly but only when reporting
    # hence the final computation.
    step_type = ST_LAST
    grad!(gradient, x)
    v = compute_extreme_point(lmo, gradient)
    primal = f(x)
    dual_gap = fast_dot(x, gradient) - fast_dot(v, gradient)
    tot_time = (time_ns() - time_start) / 1.0e9
    gamma = perform_line_search(
        line_search,
        t,
        f,
        grad!,
        gradient,
        x,
        d,
        1.0,
        linesearch_workspace,
        memory_mode,
    )
    if callback !== nothing
        state = CallbackState(
            t,
            primal,
            primal - dual_gap,
            dual_gap,
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
        callback(state)
    end
    return (x=x, v=v, primal=primal, dual_gap=dual_gap, traj_data=traj_data)
end

"""
    stochastic_frank_wolfe(f::StochasticObjective, lmo, x0; ...)

Stochastic version of Frank-Wolfe, evaluates the objective and gradient stochastically,
implemented through the [`FrankWolfe.StochasticObjective`](@ref) interface.

Keyword arguments include `batch_size` to pass a fixed `batch_size`
or a `batch_iterator` implementing
`batch_size = FrankWolfe.batchsize_iterate(batch_iterator)` for algorithms like
Variance-reduced and projection-free stochastic optimization, E Hazan, H Luo, 2016.

Similarly, a constant `momentum` can be passed or replaced by a `momentum_iterator`
implementing `momentum = FrankWolfe.momentum_iterate(momentum_iterator)`.
"""
function stochastic_frank_wolfe(
    f::StochasticObjective,
    lmo,
    x0;
    line_search::LineSearchMethod=Nonconvex(),
    momentum_iterator=nothing,
    momentum=nothing,
    epsilon=1e-7,
    max_iteration=10000,
    print_iter=1000,
    trajectory=false,
    verbose=false,
    memory_mode::MemoryEmphasis=InplaceEmphasis(),
    rng=Random.GLOBAL_RNG,
    batch_size=length(f.xs) ÷ 10 + 1,
    batch_iterator=nothing,
    full_evaluation=false,
    callback=nothing,
    traj_data=[],
    timeout=Inf,
    linesearch_workspace=nothing,
)

    # format string for output of the algorithm
    format_string = "%6s %13s %14e %14e %14e %14e %14e %6i\n"
    headers = ("Type", "Iteration", "Primal", "Dual", "Dual Gap", "Time", "It/sec", "Batch")

    function format_state(state, batch_size)
        rep = (
            steptype_string[Symbol(state.step_type)],
            string(state.t),
            Float64(state.primal),
            Float64(state.primal - state.dual_gap),
            Float64(state.dual_gap),
            state.time,
            state.t / state.time,
            batch_size,
        )
        return rep
    end

    t = 0
    dual_gap = Inf
    primal = Inf
    v = []
    x = x0
    d = similar(x)
    step_type = ST_REGULAR

    if trajectory
        callback = make_trajectory_callback(callback, traj_data)
    end

    if verbose
        callback = make_print_callback(callback, print_iter, headers, format_string, format_state)
    end

    time_start = time_ns()

    if line_search == Shortstep && L == Inf
        println("FATAL: Lipschitz constant not set. Prepare to blow up spectacularly.")
    end

    if line_search == FixedStep && gamma0 == 0
        println("FATAL: gamma0 not set. We are not going to move a single bit.")
    end
    if momentum_iterator === nothing && momentum !== nothing
        momentum_iterator = ConstantMomentumIterator(momentum)
    end
    if batch_iterator === nothing
        batch_iterator = ConstantBatchIterator(batch_size)
    end

    if verbose
        println("\nStochastic Frank-Wolfe Algorithm.")
        NumType = eltype(x0)
        println(
            "MEMORY_MODE: $memory_mode STEPSIZE: $line_search EPSILON: $epsilon max_iteration: $max_iteration TYPE: $NumType",
        )
        println(
            "GRADIENTTYPE: $(typeof(f.storage)) MOMENTUM: $(momentum_iterator !== nothing) batch policy: $(typeof(batch_iterator)) ",
        )
        println("LMO: $(typeof(lmo))")
        if memory_mode isa InplaceEmphasis
            @info("In memory_mode memory iterates are written back into x0!")
        end
    end

    if memory_mode isa InplaceEmphasis && !isa(x, Union{Array,SparseArrays.AbstractSparseArray})
        if eltype(x) <: Integer
            x = copyto!(similar(x, float(eltype(x))), x)
        else
            x = copyto!(similar(x), x)
        end
    end
    first_iter = true
    gradient = 0
    if linesearch_workspace === nothing
        linesearch_workspace = build_linesearch_workspace(line_search, x, gradient)
    end

    while t <= max_iteration && dual_gap >= max(epsilon, eps(float(typeof(dual_gap))))

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
        batch_size = batchsize_iterate(batch_iterator)

        if momentum_iterator === nothing
            gradient = compute_gradient(
                f,
                x,
                rng=rng,
                batch_size=batch_size,
                full_evaluation=full_evaluation,
            )
        elseif first_iter
            gradient = copy(
                compute_gradient(
                    f,
                    x,
                    rng=rng,
                    batch_size=batch_size,
                    full_evaluation=full_evaluation,
                ),
            )
        else
            momentum = momentum_iterate(momentum_iterator)
            compute_gradient(f, x, rng=rng, batch_size=batch_size, full_evaluation=full_evaluation)
            # gradient = momentum * gradient + (1 - momentum) * f.storage
            LinearAlgebra.mul!(gradient, LinearAlgebra.I, f.storage, 1 - momentum, momentum)
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

        d = muladd_memory_mode(memory_mode, d, x, v)

        # note: only linesearch methods that do not require full evaluations are supported
        # so nothing is passed as function
        gamma = perform_line_search(
            line_search,
            t,
            nothing,
            nothing,
            gradient,
            x,
            d,
            1.0,
            linesearch_workspace,
            memory_mode,
        )
        t += 1
        if callback !== nothing
            state = CallbackState(
                t,
                primal,
                primal - dual_gap,
                dual_gap,
                tot_time,
                x,
                v,
                d,
                gamma,
                f,
                nothing,
                lmo,
                gradient,
                step_type,
            )
            if callback(state, batch_size) === false
                break
            end
        end

        x = muladd_memory_mode(memory_mode, x, gamma, d)
    end
    # recompute everything once for final verfication / no additional callback call
    # this is important as some variants do not recompute f(x) and the dual_gap regularly but only when reporting
    # hence the final computation.
    # last computation done with full evaluation for exact gradient

    (primal, gradient) = compute_value_gradient(f, x, full_evaluation=true)
    v = compute_extreme_point(lmo, gradient)
    # @show (gradient, primal)
    dual_gap = fast_dot(x, gradient) - fast_dot(v, gradient)
    step_type = ST_LAST
    d = muladd_memory_mode(memory_mode, d, x, v)
    gamma = perform_line_search(
        line_search,
        t,
        nothing,
        nothing,
        gradient,
        x,
        d,
        1.0,
        linesearch_workspace,
        memory_mode,
    )
    tot_time = (time_ns() - time_start) / 1e9
    if callback !== nothing
        state = CallbackState(
            t,
            primal,
            primal - dual_gap,
            dual_gap,
            (time_ns() - time_start) / 1e9,
            x,
            v,
            d,
            gamma,
            f,
            nothing,
            lmo,
            gradient,
            step_type,
        )
        callback(state, batch_size)
    end
    return (x=x, v=v, primal=primal, dual_gap=dual_gap, traj_data=traj_data)
end
