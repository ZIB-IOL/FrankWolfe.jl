# Alternating Linear Minimization with a start direction instead of an initial point x0
# The is for the case of unknown feasible points.
function alternating_linear_minimization(
    bc_method,
    f,
    grad!,
    lmos::NTuple{N,LinearMinimizationOracle},
    start_direction::T;
    lambda=1.0,
    verbose=false,
    callback=nothing,
    print_iter=1e3,
    kwargs...,
) where {N,T<:AbstractArray}

    x0 = compute_extreme_point(ProductLMO(lmos), tuple(fill(start_direction, N)...))

    return alternating_linear_minimization(
        bc_method,
        f,
        grad!,
        lmos,
        x0;
        lambda=lambda,
        verbose=verbose,
        callback=callback,
        print_iter=print_iter,
        kwargs...,
    )

end

"""
    alternating_linear_minimization(bc_algo::BlockCoordinateMethod, f, grad!, lmos::NTuple{N,LinearMinimizationOracle}, x0; ...) where {N}

Alternating Linear Minimization minimizes the objective `f` over the intersections of the feasible domains specified by `lmos`.
The tuple `x0` defines the initial points for each domain.
Returns a tuple `(x, v, primal, dual_gap, infeas, traj_data)` with:
- `x` cartesian product of final iterates
- `v` cartesian product of last vertices of the LMOs
- `primal` primal value `f(x)`
- `dual_gap` final Frank-Wolfe gap
- `infeas` sum of squared, pairwise distances between iterates
- `traj_data` vector of trajectory information.
"""
function alternating_linear_minimization(
    bc_method,
    f,
    grad!,
    lmos::NTuple{N,LinearMinimizationOracle},
    x0::Tuple{Vararg{Any,N}};
    lambda=1.0,
    verbose=false,
    trajectory=false,
    callback=nothing,
    max_iteration=10000,
    print_iter = max_iteration / 10,
    memory_mode=InplaceEmphasis(),
    line_search::LS=Adaptive(),
    epsilon=1e-7,
    momentum=nothing,
    kwargs...,
) where {N, LS<:Union{LineSearchMethod,NTuple{N,LineSearchMethod}}}

    x0_bc = BlockVector([x0[i] for i in 1:N], [size(x0[i]) for i in 1:N], sum(length, x0))
    gradf = similar(x0_bc)
    prod_lmo = ProductLMO(lmos)

    function grad_bc!(storage, x)
        for i in 1:N
            grad!(gradf.blocks[i], x.blocks[i])
        end
        t = [lambda * 2.0 * (N * b - sum(x.blocks)) for b in x.blocks]
        return storage.blocks = gradf.blocks + t
    end

    dist2(x) = sum(
        fast_dot(x.blocks[i] - x.blocks[j], x.blocks[i] - x.blocks[j]) for i in 1:N for j in 1:i-1
    )

    f_bc(x) = sum(f(x.blocks[i]) for i in 1:N) + lambda * dist2(x)

    gradient = similar(x0_bc)
    grad_bc!(gradient, x0_bc)
    @info("Initialisation", x0_bc, gradient)

    dist2_data = []
    if trajectory
        function make_dist2_callback(callback)
            return function callback_dist2(state, args...)
                push!(dist2_data, dist2(state.x))
                if callback === nothing
                    return true
                end
                return callback(state, args...)
            end
        end

        callback = make_dist2_callback(callback)
    end

    if verbose
        println("\nAlternating Linear Minimization (ALM).")
        println("FW METHOD: $bc_method")

        num_type = eltype(x0[1])
        grad_type = eltype(gradf.blocks[1])
        line_search_type = line_search isa Tuple ? [typeof(a) for a in line_search] : typeof(line_search)
        println("MEMORY_MODE: $memory_mode STEPSIZE: $line_search_type EPSILON: $epsilon MAXITERATION: $max_iteration")
        println("TYPE: $num_type GRADIENTTYPE: $grad_type")
        println("MOMENTUM: $momentum LAMBDA: $lambda")

        if memory_mode isa InplaceEmphasis
            @info("In memory_mode memory iterates are written back into x0!")
        end

        # header and format string for output of the algorithm
        headers = ["Type", "Iteration", "Primal", "Dual", "Dual Gap", "Time", "It/sec", "Dist2"]
        format_string = "%6s %13s %14e %14e %14e %14e %14e %14e\n"

        function format_state(state, args...)
            rep = (
                st[Symbol(state.tt)],
                string(state.t),
                Float64(state.primal),
                Float64(state.primal - state.dual_gap),
                Float64(state.dual_gap),
                state.time,
                state.t / state.time,
                Float64(dist2(state.x)),
            )

            if bc_method in
               [away_frank_wolfe, blended_pairwise_conditional_gradient, pairwise_frank_wolfe]
                add_rep = (length(args[1]))
            elseif bc_method === blended_conditional_gradient
                add_rep = (length(args[1]), args[2])
            elseif bc_method === stochastic_frank_wolfe
                add_rep = (args[1],)
            else
                add_rep = ()
            end

            return (rep..., add_rep...)
        end

        if bc_method in
           [away_frank_wolfe, blended_pairwise_conditional_gradient, pairwise_frank_wolfe]
            push!(headers, "#ActiveSet")
            format_string = format_string[1:end-1] * " %14i\n"
        elseif bc_method === blended_conditional_gradient
            append!(headers, ["#ActiveSet", "#non-simplex"])
            format_string = format_string[1:end-1] * " %14i %14i\n"
        elseif bc_method === stochastic_frank_wolfe
            push!(headers, "Batch")
            format_string = format_string[1:end-1] * " %6i\n"
        end

        callback = make_print_callback(callback, print_iter, headers, format_string, format_state)
    end

    x, v, primal, dual_gap, traj_data = bc_method(
        f_bc,
        grad_bc!,
        prod_lmo,
        x0_bc;
        verbose=false, # Suppress inner verbose output
        trajectory=trajectory,
        callback=callback,
        print_iter=print_iter,
        kwargs...,
    )

    if trajectory
        traj_data = [(t..., dist2_data[i]) for (i, t) in enumerate(traj_data)]
    end
    return x, v, primal, dual_gap, dist2(x), traj_data
end


function ProjectionFW(y, lmo; max_iter=10000, eps=1e-3)
    f(x) = sum(abs2, x - y)
    grad!(storage, x) = storage .= 2 * (x - y)

    x0 = FrankWolfe.compute_extreme_point(lmo, y)
    x_opt, _ = FrankWolfe.frank_wolfe(
        f,
        grad!,
        lmo,
        x0,
        epsilon=eps,
        max_iteration=max_iter,
        trajectory=true,
        line_search=FrankWolfe.Adaptive(verbose=false, relaxed_smoothness=false),
    )
    return x_opt
end

"""
    alternating_projections(lmos::NTuple{N,LinearMinimizationOracle}, x0; ...) where {N}

Computes a point in the intersection of feasible domains specified by `lmos`.
Returns a tuple `(x, v, dual_gap, infeas, traj_data)` with:
- `x` cartesian product of final iterates
- `v` cartesian product of last vertices of the LMOs
- `dual_gap` final Frank-Wolfe gap
- `infeas` sum of squared, pairwise distances between iterates
- `traj_data` vector of trajectory information.
"""
function alternating_projections(
    lmos::NTuple{N,LinearMinimizationOracle},
    x0;
    epsilon=1e-7,
    max_iteration=10000,
    print_iter=1000,
    trajectory=false,
    verbose=false,
    memory_mode::MemoryEmphasis=InplaceEmphasis(),
    callback=nothing,
    traj_data=[],
    timeout=Inf,
) where {N}
    return alternating_projections(
        ProductLMO(lmos),
        x0;
        epsilon,
        max_iteration,
        print_iter,
        trajectory,
        verbose,
        memory_mode,
        callback,
        traj_data,
        timeout,
    )
end


function alternating_projections(
    lmo::ProductLMO{N},
    x0;
    epsilon=1e-7,
    max_iteration=10000,
    print_iter=1000,
    trajectory=false,
    verbose=false,
    memory_mode::MemoryEmphasis=InplaceEmphasis(),
    callback=nothing,
    traj_data=[],
    timeout=Inf,
) where {N}

    # header and format string for output of the algorithm
    headers = ["Type", "Iteration", "Dual Gap", "Infeas", "Time", "It/sec"]
    format_string = "%6s %13s %14e %14e %14e %14e\n"
    function format_state(state, infeas)
        rep = (
            st[Symbol(state.tt)],
            string(state.t),
            Float64(state.dual_gap),
            Float64(infeas),
            state.time,
            state.t / state.time,
        )
        return rep
    end

    t = 0
    dual_gap = Inf
    x = fill(x0, N)
    v = similar(x)
    tt = regular
    gradient = similar(x)
    ndim = ndims(x)

    infeasibility(x) = sum(
        fast_dot(
            selectdim(x, ndim, i) - selectdim(x, ndim, j),
            selectdim(x, ndim, i) - selectdim(x, ndim, j),
        ) for i in 1:N for j in 1:i-1
    )

    partial_infeasibility(x) =
        sum(fast_dot(x[mod(i - 2, N)+1] - x[i], x[mod(i - 2, N)+1] - x[i]) for i in 1:N)

    function grad!(storage, x)
        @. storage = [2 * (x[i] - x[mod(i - 2, N)+1]) for i in 1:N]
    end

    projection_step(x, i, t) = ProjectionFW(x, lmo.lmos[i]; eps=1 / (t^2 + 1))


    if trajectory
        callback = make_trajectory_callback(callback, traj_data)
    end

    if verbose
        callback = make_print_callback(callback, print_iter, headers, format_string, format_state)
    end

    time_start = time_ns()

    if verbose
        println("\nAlternating Projections.")
        num_type = eltype(x0[1])
        println(
            "MEMORY_MODE: $memory_mode EPSILON: $epsilon MAXITERATION: $max_iteration TYPE: $num_type",
        )
        grad_type = typeof(gradient)
        println("GRADIENTTYPE: $grad_type")
        if memory_mode isa InplaceEmphasis
            @info("In memory_mode memory iterates are written back into x0!")
        end
    end

    first_iter = true

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

        # Projection step:
        for i in 1:N
            # project the previous iterate on the i-th feasible region
            x[i] = projection_step(x[mod(i - 2, N)+1], i, t)
        end

        # Update gradients
        grad!(gradient, x)

        # Update dual gaps
        v = compute_extreme_point.(lmo.lmos, gradient)
        dual_gap = fast_dot(x - v, gradient)

        # go easy on the memory - only compute if really needed
        if ((mod(t, print_iter) == 0 && verbose) || callback !== nothing)
            infeas = infeasibility(x)
        end

        first_iter = false

        t = t + 1
        if callback !== nothing
            state = CallbackState(
                t,
                infeas,
                infeas - dual_gap,
                dual_gap,
                tot_time,
                x,
                v,
                nothing,
                nothing,
                nothing,
                nothing,
                lmo,
                gradient,
                tt,
            )
            # @show state
            if callback(state, infeas) === false
                break
            end
        end


    end
    # recompute everything once for final verfication / do not record to trajectory though for now!
    # this is important as some variants do not recompute f(x) and the dual_gap regularly but only when reporting
    # hence the final computation.
    tt = last
    infeas = infeasibility(x)
    grad!(gradient, x)
    v = compute_extreme_point.(lmo.lmos, gradient)
    dual_gap = fast_dot(x - v, gradient)

    tot_time = (time_ns() - time_start) / 1.0e9

    if callback !== nothing
        state = CallbackState(
            t,
            infeas,
            infeas - dual_gap,
            dual_gap,
            tot_time,
            x,
            v,
            nothing,
            nothing,
            nothing,
            nothing,
            lmo,
            gradient,
            tt,
        )
        callback(state, infeas)
    end

    return x, v, dual_gap, infeas, traj_data

end
