
## interface functions for LMOs that are supported by the decomposition-invariant algorithm

"""
    is_decomposition_invariant_oracle(lmo)

Function to indicate whether the given LMO supports the decomposition-invariant interface.
This interface includes `compute_extreme_point` with a `lazy` keyword, `compute_inface_extreme_point`
and `dicg_maximum_step`.
"""
is_decomposition_invariant_oracle(::LinearMinimizationOracle) = false

"""
    compute_inface_extreme_point(lmo, direction, x; lazy, kwargs...)

LMO-like operation which computes a vertex minimizing in `direction` on the face defined by the current fixings.
Fixings are maintained by the oracle (or deduced from `x` itself).
"""
compute_inface_extreme_point(lmo, direction, x; lazy, kwargs...)

"""
    dicg_maximum_step(lmo, x, direction)

Given `x` the current iterate and `direction` the negative of the direction towards which the iterate will move,
determine a maximum step size `gamma_max`, such that `x - gamma_max * direction` is in the polytope.
"""
dicg_maximum_step(lmo, x, direction)

"""
    decomposition_invariant_conditional_gradient(f, grad!, lmo, x0; kwargs...)

Implements the Decomposition-Invariant Conditional Gradient from:
Garber, Ofer (2016), Linear-memory and decomposition-invariant linearly convergent conditional gradient algorithm for structured polytopes.
The algorithm performs pairwise steps with the away direction computed by calls to a modified linear oracle, see [`FrankWolfe.is_decomposition_invariant_oracle`](@ref) for the extended linear minimization oracle interface required.
"""
function decomposition_invariant_conditional_gradient(
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
    lazy=false,
    linesearch_workspace=nothing,
    lazy_tolerance=2.0,
    num_atoms=100,
)

    if !is_decomposition_invariant_oracle(lmo)
        error(
            "The provided LMO of type $(typeof(lmo)) does not support the decomposition-invariant interface",
        )
    end
    # format string for output of the algorithm
    format_string = "%6s %13s %14e %14e %14e %14e %14e\n"
    headers = ("Type", "Iteration", "Primal", "Dual", "Dual Gap", "Time", "It/sec")
    function format_state(state, args...)
        rep = (
            st[Symbol(state.tt)],
            string(state.t),
            Float64(state.primal),
            Float64(state.primal - state.dual_gap),
            Float64(state.dual_gap),
            state.time,
            state.t / state.time,
        )
        return rep
    end

    if trajectory
        callback = make_trajectory_callback(callback, traj_data)
    end

    if verbose
        callback = make_print_callback(callback, print_iter, headers, format_string, format_state)
    end

    x = x0
    active_set = ActiveSet([(1.0, x)])
    if memory_mode isa InplaceEmphasis && !isa(x, Union{Array,SparseArrays.AbstractSparseArray})
        # if integer, convert element type to most appropriate float
        if eltype(x) <: Integer
            x = copyto!(similar(x, float(eltype(x))), x)
        else
            x = copyto!(similar(x), x)
        end
    end

    t = 0
    primal = convert(float(eltype(x)), Inf)
    tt = regular
    time_start = time_ns()

    d = similar(x)

    if gradient === nothing
        gradient = collect(x)
    end

    if verbose
        println("\nDecomposition-Invariant Conditional Gradient Algorithm.")
        NumType = eltype(x0)
        println(
            "MEMORY_MODE: $memory_mode STEPSIZE: $line_search EPSILON: $epsilon MAXITERATION: $max_iteration TYPE: $NumType",
        )
        grad_type = typeof(gradient)
        println("GRADIENTTYPE: $grad_type LAZY: $lazy lazy_tolerance: $lazy_tolerance")
        println("LMO: $(typeof(lmo))")
        if memory_mode isa InplaceEmphasis
            @info("In memory_mode memory iterates are written back into x0!")
        end
    end

    grad!(gradient, x)
    v = x0
    phi = primal
    gamma = one(phi)

    if linesearch_workspace === nothing
        linesearch_workspace = build_linesearch_workspace(line_search, x, gradient)
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
        primal = f(x)
        if t > 1
            grad!(gradient, x)
        end

        if lazy
            # error("not implemented yet")
            # _, v_local, v_local_loc, _, a_lambda, a, a_loc, _, _ =
            #     active_set_argminmax(active_set, gradient)

            # dot_forward_vertex = fast_dot(gradient, v_local)
            # dot_away_vertex = fast_dot(gradient, a)
            # local_gap = dot_away_vertex - dot_forward_vertex
            d, v, v_index, a, away_index, gamma_max, phi, tt = 
                lazy_dicg_step(
                    x, 
                    gradient, 
                    lmo, 
                    active_set, 
                    phi, 
                    epsilon, 
                    d;
                )
            if gamma_max === nothing
                gamma_max = dicg_maximum_step(lmo, x, d)
            end
        else # non-lazy, call the simple and modified
            v = compute_extreme_point(lmo, gradient, lazy=lazy)
            dual_gap = fast_dot(gradient, x) - fast_dot(gradient, v)
            phi = dual_gap
            a = compute_inface_extreme_point(lmo, NegatingArray(gradient), x; lazy=lazy)
            d = muladd_memory_mode(memory_mode, d, a, v)
            gamma_max = dicg_maximum_step(lmo, x, d)
        end
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

        if lazy
            dicg_active_set_update!(
                active_set, 
                gamma, 
                a, 
                v, 
                false, 
                away_index, 
                v_index;
                maximum_num_atoms=num_atoms, 
            )
        end
        if callback !== nothing
            state = CallbackState(
                t,
                primal,
                primal - phi,
                phi,
                tot_time,
                x,
                v,
                d,
                gamma,
                f,
                grad!,
                lmo,
                gradient,
                tt,
            )
            if callback(state) === false
                break
            end
        end
        x = muladd_memory_mode(memory_mode, x, gamma, d)
    end

    # recompute everything once more for final verfication / do not record to trajectory though
    # this is important as some variants do not recompute f(x) and the dual_gap regularly but only when reporting
    # hence the final computation.
    # do also cleanup of active_set due to many operations on the same set

    grad!(gradient, x)
    v = compute_extreme_point(lmo, gradient)
    primal = f(x)
    dual_gap = fast_dot(x, gradient) - fast_dot(v, gradient)
    if verbose
        tt = last
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
                tt,
            )
            callback(state)
        end
    end
    return (x=x, v=v, primal=primal, dual_gap=dual_gap, traj_data=traj_data)
end

"""
    blended_decomposition_invariant_conditional_gradient(f, grad!, lmo, x0; kwargs...)

Implements the Blended variant of the Decomposition-Invariant Conditional Gradient.
The algorithm performs pairwise steps with the away direction computed by calls to a modified linear oracle, see [`FrankWolfe.is_decomposition_invariant_oracle`](@ref) for the extended linear minimization oracle interface required.
"""
function blended_decomposition_invariant_conditional_gradient(
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
    lazy=false,
    linesearch_workspace=nothing,
    lazy_tolerance=2.0,
)

    if !is_decomposition_invariant_oracle(lmo)
        error(
            "The provided LMO of type $(typeof(lmo)) does not support the decomposition-invariant interface",
        )
    end
    # format string for output of the algorithm
    format_string = "%6s %13s %14e %14e %14e %14e %14e\n"
    headers = ("Type", "Iteration", "Primal", "Dual", "Dual Gap", "Time", "It/sec")
    function format_state(state, args...)
        rep = (
            st[Symbol(state.tt)],
            string(state.t),
            Float64(state.primal),
            Float64(state.primal - state.dual_gap),
            Float64(state.dual_gap),
            state.time,
            state.t / state.time,
        )
        return rep
    end

    if trajectory
        callback = make_trajectory_callback(callback, traj_data)
    end

    if verbose
        callback = make_print_callback(callback, print_iter, headers, format_string, format_state)
    end

    x = x0
    if memory_mode isa InplaceEmphasis && !isa(x, Union{Array,SparseArrays.AbstractSparseArray})
        # if integer, convert element type to most appropriate float
        if eltype(x) <: Integer
            x = copyto!(similar(x, float(eltype(x))), x)
        else
            x = copyto!(similar(x), x)
        end
    end

    t = 0
    primal = convert(float(eltype(x)), Inf)
    tt = regular
    time_start = time_ns()

    d = similar(x)

    if gradient === nothing
        gradient = collect(x)
    end

    if verbose
        println("\nBlended Decomposition-Invariant Conditional Gradient Algorithm.")
        NumType = eltype(x0)
        println(
            "MEMORY_MODE: $memory_mode STEPSIZE: $line_search EPSILON: $epsilon MAXITERATION: $max_iteration TYPE: $NumType",
        )
        grad_type = typeof(gradient)
        println("GRADIENTTYPE: $grad_type LAZY: $lazy lazy_tolerance: $lazy_tolerance")
        println("LMO: $(typeof(lmo))")
        if memory_mode isa InplaceEmphasis
            @info("In memory_mode memory iterates are written back into x0!")
        end
    end

    grad!(gradient, x)
    v = x0
    phi = primal
    gamma = one(phi)

    if linesearch_workspace === nothing
        linesearch_workspace = build_linesearch_workspace(line_search, x, gradient)
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

        primal = f(x)
        if t > 1
            grad!(gradient, x)
        end

        if lazy
            error("not implemented yet")
        else # non-lazy, call the simple and modified
            a = compute_inface_extreme_point(lmo, NegatingArray(gradient), x; lazy=lazy)
            v_inface = compute_inface_extreme_point(lmo, gradient, x; lazy=lazy)
            v = compute_extreme_point(lmo, gradient, lazy=lazy)
            inface_gap = dot(gradient, a) - fast_dot(gradient, v_inface)
            dual_gap = fast_dot(gradient, x) - fast_dot(gradient, v)
            phi = dual_gap
            # in-face step
            if inface_gap >= phi / lazy_tolerance
                tt = pairwise
                d = muladd_memory_mode(memory_mode, d, a, v)
                gamma_max = dicg_maximum_step(lmo, x, d)
            else # global FW step
                tt = regular
                d = muladd_memory_mode(memory_mode, d, x, v)
                gamma_max = one(phi)
            end
        end
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
        if callback !== nothing
            state = CallbackState(
                t,
                primal,
                primal - phi,
                phi,
                tot_time,
                x,
                v,
                d,
                gamma,
                f,
                grad!,
                lmo,
                gradient,
                tt,
            )
            if callback(state) === false
                break
            end
        end
        x = muladd_memory_mode(memory_mode, x, gamma, d)
    end

    # recompute everything once more for final verfication / do not record to trajectory though
    # this is important as some variants do not recompute f(x) and the dual_gap regularly but only when reporting
    # hence the final computation.
    # do also cleanup of active_set due to many operations on the same set

    grad!(gradient, x)
    v = compute_extreme_point(lmo, gradient)
    primal = f(x)
    dual_gap = fast_dot(x, gradient) - fast_dot(v, gradient)
    if verbose
        tt = last
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
                tt,
            )
            callback(state)
        end
    end
    return (x=x, v=v, primal=primal, dual_gap=dual_gap, traj_data=traj_data)
end

function lazy_dicg_step(
    x,
    gradient,
    lmo,
    active_set,
    phi,
    epsilon,
    d;
    use_extra_vertex_storage=false,
    extra_vertex_storage=nothing,
    lazy_tolerance=2.0,
    memory_mode::MemoryEmphasis=InplaceEmphasis(),
)
    _, v_local, v_local_loc, _, a_lambda, a_local, a_local_loc, _, _ =
        active_set_argminmax(active_set, gradient)
    # We will always have an away vertex determining the steplength. 
    tt = regular
    gamma_max = nothing
    away_index = nothing
    fw_index = nothing
    grad_dot_x = fast_dot(x, gradient)
    grad_dot_a_local = fast_dot(a_local, gradient)

    # Do lazy pairwise step
    grad_dot_lazy_fw_vertex = fast_dot(v_local, gradient)

    if grad_dot_a_local - grad_dot_lazy_fw_vertex >= phi / lazy_tolerance &&
       grad_dot_a_local - grad_dot_lazy_fw_vertex >= epsilon
        tt = lazy
        v = v_local
        a = a_local
        gamma_max = a_lambda
        d = muladd_memory_mode(memory_mode, d, a, v)
        fw_index = v_local_loc
    else
        # optionally: try vertex storage
        if use_extra_vertex_storage
            lazy_threshold = fast_dot(gradient, a_local) - phi / lazy_tolerance
            (found_better_vertex, new_forward_vertex) =
                storage_find_argmin_vertex(extra_vertex_storage, gradient, lazy_threshold)
            if found_better_vertex
                @debug("Found acceptable lazy vertex in storage")
                v = new_forward_vertex
                tt = lazylazy
            else
                v = compute_extreme_point(lmo, gradient)
                tt = pairwise
            end
        else
            v = compute_extreme_point(lmo, gradient)
            grad_dot_v = fast_dot(gradient, v)
            # Do lazy inface_point
            if grad_dot_a_local - grad_dot_v >= phi / lazy_tolerance && 
                grad_dot_a_local - grad_dot_v >= epsilon
                tt = lazy_pairwise
                a = a_local
                gamma_max = a_lambda
                away_index = a_local_loc
            else
                a = compute_inface_extreme_point(lmo, NegatingArray(gradient), x)
            end
        end

        # Real dual gap promises enough progress.
        grad_dot_fw_vertex = fast_dot(v, gradient)
        dual_gap = grad_dot_x - grad_dot_fw_vertex
        
        if dual_gap >= phi / lazy_tolerance
            d = muladd_memory_mode(memory_mode, d, a, v)
            #Lower our expectation for progress.
        else
            d = muladd_memory_mode(memory_mode, d, a, v)
            phi = min(dual_gap, phi / 2.0)
        end
    end
    return d, v, fw_index, a, away_index, gamma_max, phi, tt
end

function dicg_active_set_update!(active_set::AbstractActiveSet, lambda, atom_a, atom_v, renorm=true, idx_a=nothing, idx_v=nothing; maximum_num_atoms=100, add_dropped_vertices=false, vertex_storage=nothing)
    if idx_a === nothing
        idx_a = find_atom(active_set, atom_a)
    end

    if idx_v === nothing
        idx_v = find_atom(active_set, atom_v)
    end

    # update the weight of away_vertex if existed in active_set
    if idx_a != -1
        active_set.weights[idx_a] -= lambda
    else # shrink all the weights to ensure feasiblity and the weights are inexact.
        active_set.weights .-= lambda
    end

    if idx_v != -1
        active_set.weights[idx_v] += lambda
    else
        push!(active_set, (lambda, atom_v))
    end
    
    active_set_cleanup!(active_set, update=false, add_dropped_vertices=false, vertex_storage=nothing)
    num_atoms = length(active_set.atoms)

    if num_atoms > maximum_num_atoms
        idx_redundant_atoms = partialsortperm(active_set.atoms, (num_atoms-maximum_num_atoms))
        deleteat!(active_set, idx_redundant_atoms)
    end
    return active_set
end