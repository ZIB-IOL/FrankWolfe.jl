"""
    alternating_linear_minimization(bc_algo::BlockCoordinateMethod, f, grad!, lmos::Tuple{LinearMinimizationOracle}, x0; ...)

Alternating Linear Minimization minimizes the objective `f` over the intersections of the feasible domains specified by `lmos`.
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
    lmos::TL,
    x0;
    lambda=1.0,
    kwargs...,
) where {N,TL<:NTuple{N,LinearMinimizationOracle}}

    ndim = ndims(x0) + 1 # New product dimension
    prod_lmo = ProductLMO(lmos)
    x0_bc = cat(compute_extreme_point(prod_lmo, tuple([x0 for i in 1:N]...))..., dims=ndim)

    # workspace for the gradient
    gradf = similar(x0_bc)

    function grad_bc!(storage, x)
        for i in 1:N
            grad!(selectdim(gradf, ndim, i), selectdim(x, ndim, i))
        end
        t = lambda * 2.0 * (N * x .- sum(x, dims=ndim))
        @. storage = gradf + t
    end

    infeasibility(x) = sum(
        fast_dot(
            selectdim(x, ndim, i) - selectdim(x, ndim, j),
            selectdim(x, ndim, i) - selectdim(x, ndim, j),
        ) for i in 1:N for j in 1:i-1
    )

    f_bc(x) = sum(f(selectdim(x, ndim, i)) for i in 1:N) + lambda * infeasibility(x)

    x, v, primal, dual_gap, traj_data = bc_method(f_bc, grad_bc!, prod_lmo, x0_bc; kwargs...)

    return x, v, primal, dual_gap, infeasibility(x), traj_data
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
        line_search=FrankWolfe.Adaptive(verbose=false, relaxed_smoothness=true),
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
    function format_state(state)
        rep = (
            st[Symbol(state.tt)],
            string(state.t),
            Float64(state.dual_gap),
            Float64(state.infeas),
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
            state = CallbackStateBlockCoordinateMethod(
                t,
                infeas,
                infeas - dual_gap,
                dual_gap,
                infeas,
                tot_time,
                x,
                v,
                nothing,
                nothing,
                nothing,
                lmo,
                gradient,
                tt,
            )
            # @show state
            if callback(state) === false
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
        state = CallbackStateBlockCoordinateMethod(
            t,
            infeas,
            infeas - dual_gap,
            dual_gap,
            infeas,
            tot_time,
            x,
            v,
            nothing,
            nothing,
            nothing,
            lmo,
            gradient,
            tt,
        )
        callback(state)
    end

    return x, v, dual_gap, infeas, traj_data

end
