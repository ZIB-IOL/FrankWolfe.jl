
"""
    blended_conditional_gradient(f, grad!, lmo, x0)

Entry point for the Blended Conditional Gradient algorithm.
See Braun, Gábor, et al. "Blended conditonal gradients" ICML 2019.
The method works on an active set like [`FrankWolfe.away_frank_wolfe`](@ref),
performing gradient descent over the convex hull of active vertices,
removing vertices when their weight drops to 0 and adding new vertices
by calling the linear oracle in a lazy fashion.
"""
function blended_conditional_gradient(
    f,
    grad!,
    lmo,
    x0;
    line_search::LineSearchMethod=Adaptive(),
    line_search_inner::LineSearchMethod=Adaptive(),
    L=Inf,
    gamma0=0,
    hessian=nothing,
    step_lim=20,
    epsilon=1e-7,
    max_iteration=10000,
    print_iter=1000,
    trajectory=false,
    verbose=false,
    linesearch_tol=1e-7,
    emphasis=nothing,
    accelerated=false,
    lazy_tolerance=2.0,
    weight_purge_threshold=1e-9,
    gradient=nothing,
    callback=nothing,
    timeout=Inf,
    print_callback=print_callback,
    lmo_kwargs...,
)

    # format string for output of the algorithm
    format_string = "%6s %13s %14e %14e %14e %14e %14e %14i %14i\n"

    t = 0
    primal = Inf
    dual_gap = Inf
    active_set = ActiveSet([(1.0, x0)])
    x = x0
    if gradient === nothing
        gradient = similar(x0)
    end
    primal = f(x)
    grad!(gradient, x)
    # initial gap estimate computation
    vmax = compute_extreme_point(lmo, gradient)
    phi = fast_dot(gradient, x0 - vmax) / 2
    dual_gap = phi
    traj_data = []
    if trajectory && callback === nothing
        callback = trajectory_callback(traj_data)
    end
    tt = regular
    time_start = time_ns()
    v = x0

    if line_search isa Shortstep && !isfinite(L)
        @error("Lipschitz constant not set to a finite value. Prepare to blow up spectacularly.")
    end

    if line_search isa Agnostic || line_search isa Nonconvex
        @error("Lazification is not known to converge with open-loop step size strategies.")
    end

    if line_search isa FixedStep && gamma0 == 0
        println("WARNING: gamma0 not set. We are not going to move a single bit.")
    end

    if verbose
        println("\nBlended Conditional Gradients Algorithm.")
        NumType = eltype(x0)
        println(
            "EMPHASIS: $memory STEPSIZE: $line_search EPSILON: $epsilon MAXITERATION: $max_iteration TYPE: $NumType",
        )
        grad_type = typeof(gradient)
        println("GRADIENTTYPE: $grad_type lazy_tolerance: $lazy_tolerance")
        println("WARNING: In memory emphasis mode iterates are written back into x0!")
        headers = (
            "Type",
            "Iteration",
            "Primal",
            "Dual",
            "Dual Gap",
            "Time",
            "It/sec",
            "#ActiveSet",
            "#non-simplex",
        )
        print_callback(headers, format_string, print_header=true)
    end
    # ensure x is a mutable type
    if !isa(x, Union{Array,SparseArrays.AbstractSparseArray})
        x = copyto!(similar(x), x)
    end
    non_simplex_iter = 0
    force_fw_step = false

    while t <= max_iteration && (phi ≥ epsilon || t == 0) # do at least one iteration for consistency with other algos

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


        # TODO replace with single call interface from function_gradient.jl
        #Mininize over the convex hull until strong Wolfe gap is below a given tolerance.
        num_simplex_descent_steps = minimize_over_convex_hull!(
            f,
            grad!,
            gradient,
            active_set::ActiveSet,
            phi,
            t,
            time_start,
            non_simplex_iter,
            line_search_inner=line_search_inner,
            verbose=verbose,
            print_iter=print_iter,
            hessian=hessian,
            accelerated=accelerated,
            max_iteration=max_iteration,
            callback=callback,
            timeout=timeout,
            print_callback=print_callback,
            format_string=format_string,
        )
        t += num_simplex_descent_steps
        #Take a FW step.
        x = get_active_set_iterate(active_set)
        primal = f(x)
        grad!(gradient, x)
        # compute new atom
        (v, value) = lp_separation_oracle(
            lmo,
            active_set,
            gradient,
            phi,
            lazy_tolerance;
            inplace_loop=(emphasis == memory),
            force_fw_step=force_fw_step,
            lmo_kwargs...,
        )
        force_fw_step = false
        xval = fast_dot(x, gradient)
        if value > xval - phi / lazy_tolerance
            tt = dualstep
            # setting gap estimate as ∇f(x) (x - v_FW) / 2
            phi = (xval - value) / 2
        else
            tt = regular
            gamma, L = line_search_wrapper(
                line_search,
                t,
                f,
                grad!,
                x,
                x - v,
                gradient,
                dual_gap,
                L,
                gamma0,
                linesearch_tol,
                step_lim,
                1.0,
            )

            if gamma == 1.0
                active_set_initialize!(active_set, v)
            else
                active_set_update!(active_set, gamma, v)
            end
        end

        x = get_active_set_iterate(active_set)
        dual_gap = phi
        if callback !== nothing
            state = (
                t=t,
                primal=primal,
                dual=primal - dual_gap,
                dual_gap=dual_gap,
                time=tot_time,
                x=x,
                v=v,
                active_set_length=length(active_set),
                non_simplex_iter=non_simplex_iter,
            )
            callback(state)
        end

        if verbose && mod(t, print_iter) == 0
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
                length(active_set),
                non_simplex_iter,
            )
            print_callback(rep, format_string)
            flush(stdout)
        end

        t = t + 1
        non_simplex_iter += 1

    end

    ## post-processing and cleanup after loop

    # report last iteration
    if verbose
        x = get_active_set_iterate(active_set)
        grad!(gradient, x)
        v = compute_extreme_point(lmo, gradient)
        primal = f(x)
        dual_gap = fast_dot(x, gradient) - fast_dot(v, gradient)
        tot_time = (time_ns() - time_start) / 1e9
        rep = (
            st[Symbol(last)],
            string(t - 1),
            Float64(primal),
            Float64(primal - dual_gap),
            Float64(dual_gap),
            tot_time,
            t / tot_time,
            length(active_set),
            non_simplex_iter,
        )
        print_callback(rep, format_string)
        flush(stdout)
    end

    # cleanup the active set, renormalize, and recompute values
    active_set_cleanup!(active_set, weight_purge_threshold=weight_purge_threshold)
    active_set_renormalize!(active_set)
    x = get_active_set_iterate(active_set)
    grad!(gradient, x)
    v = compute_extreme_point(lmo, gradient)
    primal = f(x)
    #dual_gap = 2phi
    dual_gap = fast_dot(x, gradient) - fast_dot(v, gradient)

    # report post-processed iteration
    if verbose
        tot_time = (time_ns() - time_start) / 1e9
        rep = (
            st[Symbol(pp)],
            string(t - 1),
            Float64(primal),
            Float64(primal - dual_gap),
            Float64(dual_gap),
            tot_time,
            t / tot_time,
            length(active_set),
            non_simplex_iter,
        )
        print_callback(rep, format_string)
        print_callback(nothing, format_string, print_footer=true)
        flush(stdout)
    end
    return x, v, primal, dual_gap, traj_data
end


"""
    minimize_over_convex_hull!

Given a function f with gradient grad! and an active set
active_set this function will minimize the function over
the convex hull of the active set until the strong-wolfe
gap over the active set is below tolerance.

It will either directly minimize over the convex hull using
simplex gradient descent, or it will transform the problem
to barycentric coordinates and minimize over the unit
probability simplex using gradient descent or Nesterov's
accelerated gradient descent.
"""
function minimize_over_convex_hull!(
    f,
    grad!,
    gradient,
    active_set::ActiveSet,
    tolerance,
    t,
    time_start,
    non_simplex_iter;
    line_search_inner=Adaptive(),
    verbose=true,
    print_iter=1000,
    hessian=nothing,
    linesearch_tol=10e-10,
    step_lim=100,
    weight_purge_threshold=1e-12,
    storage=nothing,
    accelerated=false,
    max_iteration,
    callback,
    timeout=Inf,
    print_callback=nothing,
    format_string=nothing,
)
    #No hessian is known, use simplex gradient descent.
    if hessian === nothing
        number_of_steps = simplex_gradient_descent_over_convex_hull(
            f,
            grad!,
            gradient,
            active_set::ActiveSet,
            tolerance,
            t,
            time_start,
            non_simplex_iter,
            line_search_inner=line_search_inner,
            verbose=verbose,
            print_iter=print_iter,
            linesearch_tol=linesearch_tol,
            step_lim=step_lim,
            weight_purge_threshold=weight_purge_threshold,
            max_iteration=max_iteration,
            callback=callback,
            timeout=timeout,
            print_callback=print_callback,
            format_string=format_string,
        )
    else
        x = get_active_set_iterate(active_set)
        grad!(gradient, x)
        #Rewrite as problem over the simplex
        M, b = build_reduced_problem(
            active_set.atoms,
            hessian,
            active_set.weights,
            gradient,
            tolerance,
        )
        #Early exit if we have detected that the strong-Wolfe gap is below the desired tolerance while building the reduced problem.
        if isnothing(M)
            return 0
        end
        #In case the matrices are DoubleFloats we need to cast them as Float64, because LinearAlgebra does not work with them.
        if eltype(M) === Double64
            converted_matrix = convert(Array{Float64}, M)
            L_reduced = eigmax(converted_matrix)
        else
            L_reduced = eigmax(M)
            #L_reduced = Arpack.eigs(M, nev=1, which=:LM)
        end
        reduced_f(y) =
            f(x) - fast_dot(gradient, x) +
            0.5 * transpose(x) * hessian * x +
            fast_dot(b, y) +
            0.5 * dot(y, M, y)
        function reduced_grad!(storage, x)
            return storage .= b + M * x
        end
        #Solve using Nesterov's AGD
        if accelerated
            if eltype(M) === Double64
                mu_reduced = eigmin(converted_matrix)
            else
                mu_reduced = eigmin(M)
            end
            if L_reduced / mu_reduced > 1.0
                new_weights, number_of_steps =
                    accelerated_simplex_gradient_descent_over_probability_simplex(
                        active_set.weights,
                        reduced_f,
                        reduced_grad!,
                        tolerance,
                        t,
                        time_start,
                        non_simplex_iter,
                        verbose=verbose,
                        print_iter=print_iter,
                        L=L_reduced,
                        mu=mu_reduced,
                        max_iteration=max_iteration,
                        callback=callback,
                        timeout=timeout,
                        print_callback=print_callback,
                        format_string=format_string,
                    )
                @. active_set.weights = new_weights
            end
        end
        if !accelerated || L_reduced / mu_reduced == 1.0
            #Solve using gradient descent.
            new_weights, number_of_steps = simplex_gradient_descent_over_probability_simplex(
                active_set.weights,
                reduced_f,
                reduced_grad!,
                tolerance,
                t,
                time_start,
                non_simplex_iter,
                verbose=verbose,
                print_iter=print_iter,
                L=L_reduced,
                max_iteration=max_iteration,
                callback=callback,
                timeout=timeout,
                print_callback=print_callback,
                format_string=format_string,
            )
            @. active_set.weights = new_weights
        end
    end
    active_set_cleanup!(active_set, weight_purge_threshold=weight_purge_threshold)
    return number_of_steps
end

"""
    build_reduced_problem(atoms::AbstractVector{<:AbstractVector}, hessian, weights, gradient, tolerance)

Given an active set formed by vectors , a (constant)
Hessian and a gradient constructs a quadratic problem
over the unit probability simplex that is equivalent to
minimizing the original function over the convex hull of the
active set. If λ are the barycentric coordinates of dimension
equal to the cardinality of the active set, the objective
function is:
    f(λ) = reduced_linear^T λ + 0.5 * λ^T reduced_hessian λ

In the case where we find that the current iterate has a strong-Wolfe
gap over the convex hull of the active set that is below the tolerance
we return nothing (as there is nothing to do).

"""
function build_reduced_problem(
    atoms::AbstractVector{<:ScaledHotVector},
    hessian,
    weights,
    gradient,
    tolerance,
)
    n = atoms[1].len
    k = length(atoms)
    reduced_linear = [fast_dot(gradient, a) for a in atoms]
    if strong_frankwolfe_gap(reduced_linear) <= tolerance
        return nothing, nothing
    end
    aux_matrix = zeros(eltype(atoms[1].active_val), n, k)
    #Compute the intermediate matrix.
    for i in 1:k
        aux_matrix[:, i] .= atoms[i].active_val * hessian[atoms[i].val_idx, :]
    end
    #Compute the final matrix.
    reduced_hessian = zeros(eltype(atoms[1].active_val), k, k)
    for i in 1:k
        reduced_hessian[:, i] .= atoms[i].active_val * aux_matrix[atoms[i].val_idx, :]
    end
    reduced_linear .-= reduced_hessian * weights
    return reduced_hessian, reduced_linear
end


function build_reduced_problem(
    atoms::AbstractVector{<:SparseArrays.AbstractSparseArray},
    hessian,
    weights,
    gradient,
    tolerance,
)
    n = length(atoms[1])
    k = length(atoms)

    reduced_linear = [fast_dot(gradient, a) for a in atoms]
    if strong_frankwolfe_gap(reduced_linear) <= tolerance
        return nothing, nothing
    end

    #Construct the matrix of vertices.
    vertex_matrix = spzeros(n, k)
    for i in 1:k
        vertex_matrix[:, i] .= atoms[i]
    end
    reduced_hessian = transpose(vertex_matrix) * hessian * vertex_matrix
    reduced_linear .-= reduced_hessian * weights
    return reduced_hessian, reduced_linear
end


function build_reduced_problem(
    atoms::AbstractVector{<:AbstractVector},
    hessian,
    weights,
    gradient,
    tolerance,
)
    n = length(atoms[1])
    k = length(atoms)

    reduced_linear = [fast_dot(gradient, a) for a in atoms]
    if strong_frankwolfe_gap(reduced_linear) <= tolerance
        return nothing, nothing
    end

    #Construct the matrix of vertices.
    vertex_matrix = zeros(n, k)
    for i in 1:k
        vertex_matrix[:, i] .= atoms[i]
    end
    reduced_hessian = transpose(vertex_matrix) * hessian * vertex_matrix
    reduced_linear .-= reduced_hessian * weights
    return reduced_hessian, reduced_linear
end

"""
Checks the strong Frank-Wolfe gap for the reduced problem.
"""
function strong_frankwolfe_gap(gradient)
    val_min = Inf
    val_max = -Inf
    for i in 1:length(gradient)
        temp_val = gradient[i]
        if temp_val < val_min
            val_min = temp_val
        end
        if temp_val > val_max
            val_max = temp_val
        end
    end
    return val_max - val_min
end

"""
    accelerated_simplex_gradient_descent_over_probability_simplex

Minimizes an objective function over the unit probability simplex
until the Strong-Wolfe gap is below tolerance using Nesterov's
accelerated gradient descent.
"""
function accelerated_simplex_gradient_descent_over_probability_simplex(
    initial_point,
    reduced_f,
    reduced_grad!,
    tolerance,
    t,
    time_start,
    non_simplex_iter;
    verbose=verbose,
    print_iter=print_iter,
    L=1.0,
    mu=1.0,
    max_iteration,
    callback,
    timeout=Inf,
    print_callback=nothing,
    format_string=nothing,
)
    number_of_steps = 0
    x = deepcopy(initial_point)
    x_old = deepcopy(initial_point)
    y = deepcopy(initial_point)
    gradient_x = similar(x)
    gradient_y = similar(x)
    d = similar(x)
    reduced_grad!(gradient_x, x)
    reduced_grad!(gradient_y, x)
    strong_wolfe_gap = strong_frankwolfe_gap_probability_simplex(gradient_x, x)
    q = mu / L
    # If the problem is close to convex, simply use the accelerated algorithm for convex objective functions.
    if mu < 1.0e-3
        alpha = 0.0
        alpha_old = 0.0
    else
        gamma = (1 - sqrt(q)) / (1 + sqrt(q))
    end
    while strong_wolfe_gap > tolerance && t + number_of_steps <= max_iteration
        @. x_old = x
        reduced_grad!(gradient_y, y)
        x = projection_simplex_sort(y .- gradient_y / L)
        if mu < 1.0e-3
            alpha_old = alpha
            alpha = 0.5 * (1 + sqrt(1 + 4 * alpha^2))
            gamma = (alpha_old - 1.0) / alpha
        end
        @. y = x + gamma * (x - x_old)
        number_of_steps += 1
        primal = reduced_f(x)
        reduced_grad!(gradient_x, x)
        strong_wolfe_gap = strong_frankwolfe_gap_probability_simplex(gradient_x, x)

        if callback !== nothing
            state = (
                t=t + number_of_steps,
                primal=primal,
                dual=primal - tolerance,
                dual_gap=tolerance,
                time=(time_ns() - time_start) / 1e9,
                x=x,
            )
            callback(state)
        end
        tt = simplex_descent
        if verbose && mod(t + number_of_steps, print_iter) == 0
            if t == 0
                tt = initial
            end
            tot_time = (time_ns() - time_start) / 1.0e9
            rep = (
                st[Symbol(tt)],
                string(t + number_of_steps),
                Float64(primal),
                Float64(primal - tolerance),
                Float64(tolerance),
                tot_time,
                t / tot_time,
                length(initial_point),
                non_simplex_iter,
            )
            print_callback(rep, format_string)
            flush(stdout)
        end
        if timeout < Inf
            tot_time = (time_ns() - time_start) / 1e9
            if tot_time ≥ timeout
                if verbose
                    @info "Time limit reached"
                end
                break
            end
        end
    end
    return x, number_of_steps
end

"""
    simplex_gradient_descent_over_probability_simplex

Minimizes an objective function over the unit probability simplex
until the Strong-Wolfe gap is below tolerance using gradient descent.
"""
function simplex_gradient_descent_over_probability_simplex(
    initial_point,
    reduced_f,
    reduced_grad!,
    tolerance,
    t,
    time_start,
    non_simplex_iter;
    verbose=verbose,
    print_iter=print_iter,
    L=1.0,
    max_iteration,
    callback,
    timeout=Inf,
    print_callback=nothing,
    format_string=nothing,
)
    number_of_steps = 0
    x = deepcopy(initial_point)
    gradient = similar(x)
    d = similar(x)
    reduced_grad!(gradient, x)
    strong_wolfe_gap = strong_frankwolfe_gap_probability_simplex(gradient, x)
    while strong_wolfe_gap > tolerance && t + number_of_steps <= max_iteration
        x = projection_simplex_sort(x .- gradient / L)
        number_of_steps = number_of_steps + 1
        primal = reduced_f(x)
        reduced_grad!(gradient, x)
        strong_wolfe_gap = strong_frankwolfe_gap_probability_simplex(gradient, x)

        if callback !== nothing
            state = (
                t=t + number_of_steps,
                primal=primal,
                dual=primal - tolerance,
                dual_gap=tolerance,
                time=(time_ns() - time_start) / 1e9,
                x=x,
            )
            callback(state)
        end
        tt = simplex_descent
        if verbose && mod(t + number_of_steps, print_iter) == 0
            if t == 0
                tt = initial
            end
            tot_time = (time_ns() - time_start) / 1.0e9
            rep = (
                st[Symbol(tt)],
                string(t + number_of_steps),
                Float64(primal),
                Float64(primal - tolerance),
                Float64(tolerance),
                tot_time,
                t / tot_time,
                length(initial_point),
                non_simplex_iter,
            )
            print_callback(rep, format_string)
            flush(stdout)
        end
        if timeout < Inf
            tot_time = (time_ns() - time_start) / 1e9
            if tot_time ≥ timeout
                if verbose
                    @info "Time limit reached"
                end
                break
            end
        end
    end
    return x, number_of_steps
end



"""
    projection_simplex_sort(x; s=1.0)

Perform a projection onto the probability simplex of radius `s`
using a sorting algorithm.
"""
function projection_simplex_sort(x; s=1.0)
    n = length(x)
    if sum(x) == s && all(>=(0.0), x)
        return x
    end
    v = x .- maximum(x)
    u = sort(v, rev=true)
    cssv = cumsum(u)
    rho = sum(u .* collect(1:1:n) .> (cssv .- s)) - 1
    theta = (cssv[rho+1] - s) / (rho + 1)
    w = clamp.(v .- theta, 0.0, Inf)
    return w
end

"""
    strong_frankwolfe_gap_probability_simplex

Compute the Strong-Wolfe gap over the unit probability simplex
given a gradient.
"""
function strong_frankwolfe_gap_probability_simplex(gradient, x)
    val_min = Inf
    val_max = -Inf
    for i in 1:length(gradient)
        if x[i] > 0
            temp_val = gradient[i]
            if temp_val < val_min
                val_min = temp_val
            end
            if temp_val > val_max
                val_max = temp_val
            end
        end
    end
    return val_max - val_min
end


"""
    simplex_gradient_descent_over_convex_hull(f, grad!, gradient, active_set, tolerance, t, time_start, non_simplex_iter)

Minimizes an objective function over the convex hull of the active set
until the Strong-Wolfe gap is below tolerance using simplex gradient
descent.
"""
function simplex_gradient_descent_over_convex_hull(
    f,
    grad!,
    gradient,
    active_set::ActiveSet,
    tolerance,
    t,
    time_start,
    non_simplex_iter;
    line_search_inner=Adaptive(),
    verbose=true,
    print_iter=1000,
    hessian=nothing,
    linesearch_tol=10e-10,
    step_lim=100,
    weight_purge_threshold=1e-12,
    max_iteration,
    callback,
    timeout=Inf,
    print_callback=nothing,
    format_string=nothing,
)
    number_of_steps = 0
    L_inner = nothing
    upgrade_accuracy_flag = false
    x = get_active_set_iterate(active_set)
    while t + number_of_steps ≤ max_iteration
        grad!(gradient, x)
        #Check if strong Wolfe gap over the convex hull is small enough.
        c = [fast_dot(gradient, a) for a in active_set.atoms]
        if maximum(c) - minimum(c) <= tolerance || t + number_of_steps ≥ max_iteration
            return number_of_steps
        end
        #Otherwise perform simplex steps until we get there.
        k = length(active_set)
        csum = sum(c)
        c .-= (csum / k)
        # name change to stay consistent with the paper, c is actually updated in-place
        d = c
        # NOTE: sometimes the direction is non-improving
        # usual suspects are floating-point errors when multiplying atoms with near-zero weights
        # in that case, inverting the sense of d
        # Computing the quantity below is the same as computing the <-\nabla f(x), direction>.
        # If <-\nabla f(x), direction>  >= 0 the direction is a descent direction.
        descent_direction_product = fast_dot(d, d) + (csum / k) * sum(d)
        @inbounds if descent_direction_product < 0
            current_iteration = t + number_of_steps
            @warn "Non-improving d ($descent_direction_product) due to numerical instability in iteration $current_iteration. Temporarily upgrading precision to BigFloat for the current iteration."
            # extended warning - we can discuss what to integrate
            # If higher accuracy is required, consider using Double64 (still quite fast) and if that does not help BigFloat (slower) as type for the numbers.
            # Alternatively, consider using AFW (with lazy = true) instead."
            bdir = big.(gradient)
            c = [fast_dot(bdir, a) for a in active_set.atoms]
            csum = sum(c)
            c .-= csum / k
            d = c
            descent_direction_product_inner = fast_dot(d, d) + (csum / k) * sum(d)
            @inbounds if descent_direction_product_inner < 0
                @warn "d non-improving in large precision, forcing FW"
                @warn "dot value: $descent_direction_product_inner"
                return number_of_steps
            end
        end

        η = eltype(d)(Inf)
        rem_idx = -1
        @inbounds for idx in eachindex(d)
            if d[idx] > 0
                max_val = active_set.weights[idx] / d[idx]
                if η > max_val
                    η = max_val
                    rem_idx = idx
                end
            end
        end
        # TODO at some point avoid materializing both x and y
        x = copy(active_set.x)
        η = max(0, η)
        @. active_set.weights -= η * d
        y = copy(compute_active_set_iterate!(active_set))
        number_of_steps += 1
        if f(x) ≥ f(y)
            active_set_cleanup!(active_set, weight_purge_threshold=weight_purge_threshold)
        else
            if line_search_inner isa Adaptive
                gamma, L_inner = adaptive_step_size(
                    f,
                    grad!,
                    gradient,
                    x,
                    x - y,
                    L_inner,
                    gamma_max=1.0,
                    upgrade_accuracy=upgrade_accuracy_flag,
                )
                #If the stepsize is that small we probably need to increase the accuracy of
                #the types we are using.
                if gamma < eps()
                    #@warn "Upgrading the accuracy of the adaptive line search."
                    L_inner = nothing
                    gamma, L_inner = adaptive_step_size(
                        f,
                        grad!,
                        gradient,
                        x,
                        x - y,
                        L_inner,
                        gamma_max=1.0,
                        upgrade_accuracy=true,
                    )
                end
            else
                gamma, _ = backtrackingLS(
                    f,
                    gradient,
                    x,
                    x - y,
                    1.0,
                    linesearch_tol=linesearch_tol,
                    step_lim=step_lim,
                )
            end
            gamma = min(1.0, gamma)
            # step back from y to x by (1 - γ) η d
            # new point is x - γ η d
            if gamma == 1.0
                active_set_cleanup!(active_set, weight_purge_threshold=weight_purge_threshold)
            else
                @. active_set.weights += η * (1 - gamma) * d
                @. active_set.x = x + gamma * (y - x)
            end
        end
        x = get_active_set_iterate(active_set)
        primal = f(x)
        dual_gap = tolerance

        if callback !== nothing
            state = (
                t=t,
                primal=primal,
                dual=primal - dual_gap,
                dual_gap=dual_gap,
                time=(time_ns() - time_start) / 1e9,
                x=x,
                active_set_length=length(active_set),
                non_simplex_iter=non_simplex_iter,
            )
            callback(state)
        end
        tt = simplex_descent
        if verbose && mod(t + number_of_steps, print_iter) == 0
            if t == 0
                tt = initial
            end
            tot_time = (time_ns() - time_start) / 1.0e9
            rep = (
                st[Symbol(tt)],
                string(t + number_of_steps),
                Float64(primal),
                Float64(primal - dual_gap),
                Float64(dual_gap),
                tot_time,
                t / tot_time,
                length(active_set),
                non_simplex_iter,
            )
            print_callback(rep, format_string)
            flush(stdout)
        end
        if timeout < Inf
            tot_time = (time_ns() - time_start) / 1e9
            if tot_time ≥ timeout
                if verbose
                    @info "Time limit reached"
                end
                break
            end
        end

    end
    return number_of_steps
end

"""
Returns either a tuple `(y, val)` with `y` an atom from the active set satisfying
the progress criterion and `val` the corresponding gap `dot(y, direction)`
or the same tuple with `y` from the LMO.

`inplace_loop` controls whether the iterate type allows in-place writes.
`kwargs` are passed on to the LMO oracle.
"""
function lp_separation_oracle(
    lmo::LinearMinimizationOracle,
    active_set::ActiveSet,
    direction,
    min_gap,
    lazy_tolerance;
    inplace_loop=false,
    force_fw_step::Bool=false,
    kwargs...,
)
    # if FW step forced, ignore active set
    if !force_fw_step
        ybest = active_set.atoms[1]
        x = active_set.weights[1] * active_set.atoms[1]
        if inplace_loop
            if !isa(x, Union{Array,SparseArrays.AbstractSparseArray})
                if x isa AbstractVector
                    x = convert(SparseVector{eltype(x)}, x)
                else
                    x = convert(SparseArrays.SparseMatrixCSC{eltype(x)}, x)
                end
            end
        end
        val_best = fast_dot(direction, ybest)
        for idx in 2:length(active_set)
            y = active_set.atoms[idx]
            if inplace_loop
                x .+= active_set.weights[idx] * y
            else
                x += active_set.weights[idx] * y
            end
            val = fast_dot(direction, y)
            if val < val_best
                val_best = val
                ybest = y
            end
        end
        xval = fast_dot(direction, x)
        if xval - val_best ≥ min_gap / lazy_tolerance
            return (ybest, val_best)
        end
    end
    # otherwise, call the LMO
    y = compute_extreme_point(lmo, direction; kwargs...)
    # don't return nothing but y, fast_dot(direction, y) / use y for step outside / and update phi as in LCG (lines 402 - 406)
    return (y, fast_dot(direction, y))
end
