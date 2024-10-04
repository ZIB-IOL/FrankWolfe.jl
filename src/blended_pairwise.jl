"""
    blended_pairwise_conditional_gradient(f, grad!, lmo, x0; kwargs...)

Implements the BPCG algorithm from [Tsuji, Tanaka, Pokutta (2021)](https://arxiv.org/abs/2110.12650).
The method uses an active set of current vertices.
Unlike away-step, it transfers weight from an away vertex to another vertex of the active set.
"""
function blended_pairwise_conditional_gradient(
    f,
    grad!,
    lmo,
    x0;
    line_search::LineSearchMethod=Adaptive(),
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
    renorm_interval=1000,
    lazy=false,
    linesearch_workspace=nothing,
    lazy_tolerance=2.0,
    weight_purge_threshold=weight_purge_threshold_default(eltype(x0)),
    extra_vertex_storage=nothing,
    add_dropped_vertices=false,
    use_extra_vertex_storage=false,
    recompute_last_vertex=true,
    squadratic=false,
    lp_solver=nothing,  # New parameter for LP solver
    sparsify=false,  # New parameter for sparsification
)
    # add the first vertex to active set from initialization
    active_set = ActiveSet([(1.0, x0)])

    return blended_pairwise_conditional_gradient(
        f,
        grad!,
        lmo,
        active_set,
        line_search=line_search,
        epsilon=epsilon,
        max_iteration=max_iteration,
        print_iter=print_iter,
        trajectory=trajectory,
        verbose=verbose,
        memory_mode=memory_mode,
        gradient=gradient,
        callback=callback,
        traj_data=traj_data,
        timeout=timeout,
        renorm_interval=renorm_interval,
        lazy=lazy,
        linesearch_workspace=linesearch_workspace,
        lazy_tolerance=lazy_tolerance,
        weight_purge_threshold=weight_purge_threshold,
        extra_vertex_storage=extra_vertex_storage,
        add_dropped_vertices=add_dropped_vertices,
        use_extra_vertex_storage=use_extra_vertex_storage,
        recompute_last_vertex=recompute_last_vertex,
        squadratic=squadratic,
        lp_solver=lp_solver,
        sparsify=sparsify,  # Propagate the sparsify parameter
    )
end

"""
    blended_pairwise_conditional_gradient(f, grad!, lmo, active_set::AbstractActiveSet; kwargs...)

Warm-starts BPCG with a pre-defined `active_set`.
"""
function blended_pairwise_conditional_gradient(
    f,
    grad!,
    lmo,
    active_set::AbstractActiveSet{AT,R};
    line_search::LineSearchMethod=Adaptive(),
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
    renorm_interval=1000,
    lazy=false,
    linesearch_workspace=nothing,
    lazy_tolerance=2.0,
    weight_purge_threshold=weight_purge_threshold_default(R),
    extra_vertex_storage=nothing,
    add_dropped_vertices=false,
    use_extra_vertex_storage=false,
    recompute_last_vertex=true,
    squadratic=false,
    lp_solver=nothing,  # New parameter for LP solver
    sparsify=false,  # New parameter for sparsification
) where {AT,R}

    # format string for output of the algorithm
    format_string = "%6s %13s %14e %14e %14e %14e %14e %14i\n"
    headers = ("Type", "Iteration", "Primal", "Dual", "Dual Gap", "Time", "It/sec", "#ActiveSet")
    function format_state(state, active_set, args...)
        rep = (
            steptype_string[Symbol(state.step_type)],
            string(state.t),
            Float64(state.primal),
            Float64(state.primal - state.dual_gap),
            Float64(state.dual_gap),
            state.time,
            state.t / state.time,
            length(active_set),
        )
        return rep
    end

    if trajectory
        callback = make_trajectory_callback(callback, traj_data)
    end

    if verbose
        callback = make_print_callback(callback, print_iter, headers, format_string, format_state)
    end

    t = 0
    next_direct_solve_interval = 10
    last_direct_solve = 0
    compute_active_set_iterate!(active_set)
    x = get_active_set_iterate(active_set)
    primal = convert(eltype(x), Inf)
    step_type = ST_REGULAR
    time_start = time_ns()

    d = similar(x)

    if gradient === nothing
        gradient = collect(x)
    end

    if squadratic || sparsify
        @assert lp_solver !== nothing "When squadratic or sparsify is true, lp_solver must be provided"
    end

    if verbose
        println("\nBlended Pairwise Conditional Gradient Algorithm.")
        NumType = eltype(x)
        println(
            "MEMORY_MODE: $memory_mode STEPSIZE: $line_search EPSILON: $epsilon MAXITERATION: $max_iteration TYPE: $NumType",
        )
        grad_type = typeof(gradient)
        println("GRADIENTTYPE: $grad_type LAZY: $lazy lazy_tolerance: $lazy_tolerance")
        println("LMO: $(typeof(lmo))")
        if lp_solver !== nothing
            println("LP SOLVER: $(lp_solver)")
        end
        if use_extra_vertex_storage && !lazy
            @info("vertex storage only used in lazy mode")
        end
        if (use_extra_vertex_storage || add_dropped_vertices) && extra_vertex_storage === nothing
            @warn(
                "use_extra_vertex_storage and add_dropped_vertices options are only usable with a extra_vertex_storage storage"
            )
        end
    end

    grad!(gradient, x)
    v = compute_extreme_point(lmo, gradient)
    # if !lazy, phi is maintained as the global dual gap
    phi = max(0, fast_dot(x, gradient) - fast_dot(v, gradient))
    local_gap = zero(phi)
    gamma = one(phi)

    if linesearch_workspace === nothing
        linesearch_workspace = build_linesearch_workspace(line_search, x, gradient)
    end

    if extra_vertex_storage === nothing
        use_extra_vertex_storage = add_dropped_vertices = false
    end

    while t <= max_iteration && phi >= max(epsilon, eps(epsilon))

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
        t += 1

        # compute current iterate from active set
        x = get_active_set_iterate(active_set)
        primal = f(x)
        if t > 1
            grad!(gradient, x)
        end

        _, v_local, v_local_loc, _, a_lambda, a, a_loc, _, _ =
            active_set_argminmax(active_set, gradient)

        dot_forward_vertex = fast_dot(gradient, v_local)
        dot_away_vertex = fast_dot(gradient, a)
        local_gap = dot_away_vertex - dot_forward_vertex
        if !lazy
            if t > 1
                v = compute_extreme_point(lmo, gradient)
                dual_gap = fast_dot(gradient, x) - fast_dot(gradient, v)
                phi = dual_gap
            end
        end
        # minor modification from original paper for improved sparsity
        # (proof follows with minor modification when estimating the step)
        if local_gap ≥ phi / lazy_tolerance
            d = muladd_memory_mode(memory_mode, d, a, v_local)
            vertex_taken = v_local
            gamma_max = a_lambda
            gamma = perform_line_search(
                line_search,
                t,
                f,
                grad!,
                gradient,
                x,
                d,
                gamma_max,
                linesearch_workspace,
                memory_mode,
            )
            gamma = min(gamma_max, gamma)
            step_type = gamma ≈ gamma_max ? ST_DROP : ST_PAIRWISE
            if callback !== nothing
                state = CallbackState(
                    t,
                    primal,
                    primal - phi,
                    phi,
                    tot_time,
                    x,
                    vertex_taken,
                    d,
                    gamma,
                    f,
                    grad!,
                    lmo,
                    gradient,
                    step_type,
                )
                if callback(state, active_set, a) === false
                    break
                end
            end
            # reached maximum of lambda -> dropping away vertex
            if gamma ≈ gamma_max
                active_set.weights[v_local_loc] += gamma
                deleteat!(active_set, a_loc)
                if add_dropped_vertices
                    push!(extra_vertex_storage, a)
                end
            else # transfer weight from away to local FW
                active_set.weights[a_loc] -= gamma
                active_set.weights[v_local_loc] += gamma
                @assert active_set_validate(active_set)
            end
            active_set_update_iterate_pairwise!(active_set.x, gamma, v_local, a)
        else # add to active set
            if lazy # otherwise, v computed above already
                # optionally try to use the storage
                if use_extra_vertex_storage
                    lazy_threshold = fast_dot(gradient, x) - phi / lazy_tolerance
                    (found_better_vertex, new_forward_vertex) =
                        storage_find_argmin_vertex(extra_vertex_storage, gradient, lazy_threshold)
                    if found_better_vertex
                        if verbose
                            @debug("Found acceptable lazy vertex in storage")
                        end
                        v = new_forward_vertex
                        step_type = ST_LAZYSTORAGE
                    else
                        v = compute_extreme_point(lmo, gradient)
                        step_type = ST_REGULAR
                    end
                else
                    # for t == 1, v is already computed before first iteration
                    if t > 1
                        v = compute_extreme_point(lmo, gradient)
                    end
                    step_type = ST_REGULAR
                end
            else # Set the correct flag step.
                step_type = ST_REGULAR
            end
            vertex_taken = v
            # short circuit for quadratic case
            if squadratic && (t == 1 || (t > last_direct_solve && t - last_direct_solve >= next_direct_solve_interval))
                next_direct_solve_interval *= 2  # Double the interval for next execution
                last_direct_solve = t  # Update the last execution round
                # Compute gradient at 0
                zero_x = zero(x)
                zero_grad = similar(gradient)
                grad!(zero_grad, zero_x)
                active_set = direct_solve(active_set, zero_grad, lp_solver)  # Pass lp_solver to direct_solve
                active_set_cleanup!(active_set; weight_purge_threshold=weight_purge_threshold)
                compute_active_set_iterate!(active_set)
                x = get_active_set_iterate(active_set)
                grad!(gradient, x)
                step_type = ST_DIRECT
            end
            dual_gap = fast_dot(gradient, x) - fast_dot(gradient, v)
            # if we are about to exit, compute dual_gap with the cleaned-up x
            if dual_gap ≤ epsilon
                active_set_renormalize!(active_set)
                active_set_cleanup!(active_set; weight_purge_threshold=weight_purge_threshold)
                compute_active_set_iterate!(active_set)
                x = get_active_set_iterate(active_set)
                grad!(gradient, x)
                dual_gap = fast_dot(gradient, x) - fast_dot(gradient, v)
            end
            # Note: In the following, we differentiate between lazy and non-lazy updates.
            # The reason is that the non-lazy version does not use phi but the lazy one heavily depends on it.
            # It is important that the phi is only updated after dropping
            # below phi / lazy_tolerance, as otherwise we simply have a "lagging" dual_gap estimate that just slows down convergence.
            # The logic is as follows:
            # - for non-lazy: we accept everything and there are no dual steps
            # - for lazy: we also accept slightly weaker vertices, those satisfying phi / lazy_tolerance
            # this should simplify the criterion.
            # DO NOT CHANGE without good reason and talk to Sebastian first for the logic behind this.
            if !lazy || dual_gap ≥ phi / lazy_tolerance
                d = muladd_memory_mode(memory_mode, d, x, v)

                gamma = perform_line_search(
                    line_search,
                    t,
                    f,
                    grad!,
                    gradient,
                    x,
                    d,
                    one(eltype(x)),
                    linesearch_workspace,
                    memory_mode,
                )
                if callback !== nothing
                    state = CallbackState(
                        t,
                        primal,
                        primal - phi,
                        phi,
                        tot_time,
                        x,
                        vertex_taken,
                        d,
                        gamma,
                        f,
                        grad!,
                        lmo,
                        gradient,
                        step_type,
                    )
                    if callback(state, active_set) === false
                        break
                    end
                end

                # dropping active set and restarting from singleton
                if gamma ≈ 1.0
                    if add_dropped_vertices
                        for vtx in active_set.atoms
                            if vtx != v
                                push!(extra_vertex_storage, vtx)
                            end
                        end
                    end
                    active_set_initialize!(active_set, v)
                else
                    renorm = mod(t, renorm_interval) == 0
                    active_set_update!(active_set, gamma, v, renorm, nothing)
                end
            else # dual step
                # set to computed dual_gap for consistency between the lazy and non-lazy run.
                # that is ok as we scale with the K = 2.0 default anyways
                # we only update the dual gap if the step was regular (not lazy from discarded set)
                if step_type != ST_LAZYSTORAGE
                    phi = dual_gap
                    @debug begin
                        @assert step_type == ST_REGULAR
                        v2 = compute_extreme_point(lmo, gradient)
                        g = dot(gradient, x - v2)
                        if abs(g - dual_gap) > 100 * sqrt(eps())
                            error("dual gap estimation error $g $dual_gap")
                        end
                    end
                else
                    @info "useless step"
                end
                step_type = ST_DUALSTEP
                if callback !== nothing
                    state = CallbackState(
                        t,
                        primal,
                        primal - phi,
                        phi,
                        tot_time,
                        x,
                        vertex_taken,
                        nothing,
                        gamma,
                        f,
                        grad!,
                        lmo,
                        gradient,
                        step_type,
                    )
                    if callback(state, active_set) === false
                        break
                    end
                end
            end
        end
        if mod(t, renorm_interval) == 0
            active_set_renormalize!(active_set)
            x = compute_active_set_iterate!(active_set)
        end
        if (
            ((mod(t, print_iter) == 0 || step_type == ST_DUALSTEP) == 0 && verbose) ||
            callback !== nothing ||
            !(line_search isa Agnostic || line_search isa Nonconvex || line_search isa FixedStep)
        )
            primal = f(x)
        end
    end

    # recompute everything once more for final verfication / do not record to trajectory though for now!
    # this is important as some variants do not recompute f(x) and the dual_gap regularly but only when reporting
    # hence the final computation.
    # do also cleanup of active_set due to many operations on the same set

    if verbose
        compute_active_set_iterate!(active_set)
        x = get_active_set_iterate(active_set)
        grad!(gradient, x)
        v = compute_extreme_point(lmo, gradient)
        primal = f(x)
        phi_new = fast_dot(x, gradient) - fast_dot(v, gradient)
        phi = phi_new < phi ? phi_new : phi
        step_type = ST_LAST
        tot_time = (time_ns() - time_start) / 1e9
        if callback !== nothing
            state = CallbackState(
                t,
                primal,
                primal - phi,
                phi,
                tot_time,
                x,
                v,
                nothing,
                gamma,
                f,
                grad!,
                lmo,
                gradient,
                step_type,
            )
            callback(state, active_set)
        end
    end
    active_set_renormalize!(active_set)
    if sparsify
        active_set = sparsify_iterate(active_set, lp_solver)
    end
    active_set_cleanup!(active_set; weight_purge_threshold=weight_purge_threshold)
    compute_active_set_iterate!(active_set)
    x = get_active_set_iterate(active_set)
    grad!(gradient, x)
    # otherwise values are maintained to last iteration
    if recompute_last_vertex
        v = compute_extreme_point(lmo, gradient)
        primal = f(x)
        dual_gap = fast_dot(x, gradient) - fast_dot(v, gradient)
    end
    step_type = ST_POSTPROCESS
    tot_time = (time_ns() - time_start) / 1e9
    if callback !== nothing
        state = CallbackState(
            t,
            primal,
            primal - dual_gap,
            dual_gap,
            tot_time,
            x,
            v,
            nothing,
            gamma,
            f,
            grad!,
            lmo,
            gradient,
            step_type,
        )
        callback(state, active_set)
    end

    return (x=x, v=v, primal=primal, dual_gap=dual_gap, traj_data=traj_data, active_set=active_set)
end

function direct_solve(active_set, gradient, lp_solver)
    # 1. Compute x0 = -1/2 * grad(0)
    x0 = -0.5 * gradient

    # 2. Generate matrix A with columns being the points in the active set
    if isempty(active_set.atoms)
        A = zeros(length(x0), 0)
    else
        # A = hcat(getindex.(active_set, 2)...) // this leads to stack overflow
        A = zeros(eltype(x0), length(x0), length(active_set))
        for (i, atom) in enumerate(active_set)
            A[:, i] = atom[2]
        end
    end
    # println(active_set)
    # println(A)
    # println(x0)

    # 3. Setup and solve the system using the provided LP solver
    n = size(A, 2)
    model = MOI.instantiate(lp_solver)
    MOI.set(model, MOI.Silent(), true)

    # Define variables
    λ = MOI.add_variables(model, n)
    for i in 1:n
        MOI.add_constraint(model, λ[i], MOI.GreaterThan(0.0))
    end

    # Add sum constraint
    sum_λ = MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(ones(n), λ), 0.0)
    MOI.add_constraint(model, sum_λ, MOI.EqualTo(1.0))

    # Add constraint A'A λ = A'x0
    Q = A'A
    b = A'x0
    # println(Q)
    # println(b)
    for i in 1:size(Q, 1)
        terms = MOI.ScalarAffineTerm{Float64}[]
        for j in 1:size(Q, 2)
            if !iszero(Q[i,j])
                push!(terms, MOI.ScalarAffineTerm(Q[i,j], λ[j]))
            end
        end
        constraint = MOI.ScalarAffineFunction(terms, 0.0)
        MOI.add_constraint(model, constraint, MOI.EqualTo(b[i]))
    end

    # Set a dummy objective (minimize sum of λ)
    dummy_objective = MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(ones(n), λ), 0.0)
    MOI.set(model, MOI.ObjectiveFunction{typeof(dummy_objective)}(), dummy_objective)
    MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)

    MOI.optimize!(model)

    # 4. Check if solve was feasible and update active set weights
    if MOI.get(model, MOI.TerminationStatus()) in [MOI.OPTIMAL, MOI.FEASIBLE_POINT]
        # @info "Direct solve successful"
        # λ_values = MOI.get.(model, MOI.VariablePrimal(), λ) # leads to stack overflow
        λ_values = Vector{Float64}(undef, n)
        for i in 1:n
            λ_values[i] = MOI.get(model, MOI.VariablePrimal(), λ[i])
        end
        new_weights = λ_values
        return ActiveSet([(new_weights[i], active_set.atoms[i]) for i in 1:n])
    else
#        @info "Direct solve failed"
        return active_set
    end
end

function sparsify_iterate(active_set, lp_solver)
    # 1. Get current iterate
    x0 = get_active_set_iterate(active_set)

    # 2. Generate matrix A with columns being the points in the active set
    if isempty(active_set.atoms)
        A = zeros(length(x0), 0)
    else
        # A = hcat(getindex.(active_set, 2)...) // this leads to stack overflow
        A = zeros(eltype(x0), length(x0), length(active_set))
        for (i, atom) in enumerate(active_set)
            A[:, i] = atom[2]
        end
    end
    # @info "Sparsification matrix A generated"

    # 3. Setup and solve the system using the provided LP solver
    n = size(A, 2)
    model = MOI.instantiate(lp_solver)
    MOI.set(model, MOI.Silent(), true)

    # Define variables
    λ = MOI.add_variables(model, n)
    for i in 1:n
        MOI.add_constraint(model, λ[i], MOI.GreaterThan(0.0))
    end

    # Add sum constraint
    sum_λ = MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(ones(n), λ), 0.0)
    MOI.add_constraint(model, sum_λ, MOI.EqualTo(1.0))

    # Add constraint A λ = x0
    Q = A
    b = x0
    for i in 1:size(Q, 1)
        terms = MOI.ScalarAffineTerm{Float64}[]
        for j in 1:size(Q, 2)
            if !iszero(Q[i,j])
                push!(terms, MOI.ScalarAffineTerm(Q[i,j], λ[j]))
            end
        end
        constraint = MOI.ScalarAffineFunction(terms, 0.0)
        MOI.add_constraint(model, constraint, MOI.EqualTo(b[i]))
    end
    # @info "Sparsification problem setup complete"
    # Set a dummy objective (minimize sum of λ)
    dummy_objective = MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(ones(n), λ), 0.0)
    MOI.set(model, MOI.ObjectiveFunction{typeof(dummy_objective)}(), dummy_objective)
    MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    # @info "Sparsification problem setup + objective set"
    MOI.optimize!(model)
    # @info "Sparsification problem solved"
    # 4. Check if solve was feasible and update active set weights
    if MOI.get(model, MOI.TerminationStatus()) in [MOI.OPTIMAL, MOI.FEASIBLE_POINT]
        @info "Sparsification successful"
        # λ_values = MOI.get.(model, MOI.VariablePrimal(), λ) # seems to lead to stack overflow // using explicit loop instead // does not seem to fix the problem
        λ_values = Vector{Float64}(undef, n)
        for i in 1:n
            λ_values[i] = MOI.get(model, MOI.VariablePrimal(), λ[i])
        end
        new_weights = λ_values
        return ActiveSet([(new_weights[i], active_set.atoms[i]) for i in 1:n])
    else
        @info "Sparsification failed"
        return active_set
    end
end
