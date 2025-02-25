

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
    dicg_maximum_step(lmo, direction, x)

Given `x` the current iterate and `direction` the negative of the direction towards which the iterate will move,
determine a maximum step size `gamma_max`, such that `x - gamma_max * direction` is in the polytope.
"""
dicg_maximum_step(lmo, direction, x)

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
    use_strong_lazy = false,
    linesearch_workspace=nothing,
    lazy_tolerance=2.0,
    extra_vertex_storage=nothing,
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
    step_type = ST_REGULAR
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
        println("GRADIENstep_typeYPE: $grad_type LAZY: $lazy lazy_tolerance: $lazy_tolerance")
        println("LMO: $(typeof(lmo))")
        if memory_mode isa InplaceEmphasis
            @info("In memory_mode memory iterates are wristep_typeen back into x0!")
        end
    end

    grad!(gradient, x)
    v = x0
    phi = primal
    gamma = one(phi)

    if lazy
        if extra_vertex_storage === nothing
            v = compute_extreme_point(lmo, gradient, lazy=lazy)
            pre_computed_set = [v]
        else
            pre_computed_set = extra_vertex_storage
        end
    end

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
            d, v, v_index, a, away_index, phi, step_type =
                lazy_standard_dicg_step(
                    x,
                    gradient,
                    lmo,
                    pre_computed_set,
                    phi,
                    epsilon,
                    d;
                    strong_lazification = use_strong_lazy,
                    lazy_tolerance = lazy_tolerance,
                )
        else # non-lazy, call the simple and modified
            v = compute_extreme_point(lmo, gradient, lazy=lazy)
            dual_gap = fast_dot(gradient, x) - fast_dot(gradient, v)
            phi = dual_gap
            a = compute_inface_extreme_point(lmo, NegatingArray(gradient), x; lazy=lazy)
	    d = muladd_memory_mode(memory_mode, d, a, v)
	    step_type = ST_PAIRWISE
        end
		
        gamma_max = dicg_maximum_step(lmo, d, x)
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
            idx = findfirst(x -> x == v, pre_computed_set)
            if idx !== nothing
                push!(pre_computed_set, v)
            end
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
                step_type,
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
        step_type = ST_LAST
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
    extra_vertex_storage = nothing,
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
    step_type = ST_REGULAR
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
        println("GRADIENstep_typeYPE: $grad_type LAZY: $lazy lazy_tolerance: $lazy_tolerance")
        println("LMO: $(typeof(lmo))")
        if memory_mode isa InplaceEmphasis
            @info("In memory_mode memory iterates are wristep_typeen back into x0!")
        end
    end

    grad!(gradient, x)
    v = x0
    phi = primal
    gamma = one(phi)

    if lazy
        if extra_vertex_storage === nothing
            v = compute_extreme_point(lmo, gradient, lazy = lazy)
            pre_computed_set = [v]
        else
            pre_computed_set = extra_vertex_storage
        end
    end
	
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
            d, v, v_index, a, away_index, phi, step_type =
                lazy_blended_dicg_step(
                    x,
                    gradient,
                    lmo,
                    pre_computed_set,
                    phi,
                    epsilon,
                    d;
                    strong_lazification = use_strong_lazy,
                    lazy_tolerance = lazy_tolerance,
                )
        else # non-lazy, call the simple and modified
            a = compute_inface_extreme_point(lmo, NegatingArray(gradient), x; lazy=lazy)
            v_inface = compute_inface_extreme_point(lmo, gradient, x; lazy=lazy)
            v = compute_extreme_point(lmo, gradient, lazy=lazy)
            inface_gap = dot(gradient, a) - fast_dot(gradient, v_inface)
            dual_gap = fast_dot(gradient, x) - fast_dot(gradient, v)
            phi = dual_gap
            # in-face step
            if inface_gap >= phi / lazy_tolerance
                step_type = ST_PAIRWISE
                d = muladd_memory_mode(memory_mode, d, a, v)
                gamma_max = dicg_maximum_step(lmo, d, x)
            else # global FW step
                step_type = ST_REGULAR
                d = muladd_memory_mode(memory_mode, d, x, v)
                gamma_max = one(phi)
            end
        end
        if step_type == ST_REGULAR
            gamma_max = one(phi)
        else
            gamma_max = dicg_maximum_step(lmo, d, x)
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
                step_type,
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
        step_type = ST_LAST
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
            callback(state)
        end
    end
    return (x=x, v=v, primal=primal, dual_gap=dual_gap, traj_data=traj_data)
end

function lazy_standard_dicg_step(
    x,
    gradient,
    lmo,
    pre_computed_set,
    phi,
    epsilon,
    d;
    strong_lazification = false,
    lazy_tolerance = 2.0,
    memory_mode::MemoryEmphasis = InplaceEmphasis(),
)
    v_local, v_local_loc, val, a_local, a_local_loc, valM =
        pre_computed_set_argminmax(lmo, pre_computed_set, gradient, x; strong_lazification = strong_lazification)
    step_type = ST_PAIRWISE
    away_index = nothing
    fw_index = nothing
    grad_dot_x = fast_dot(x, gradient)
    grad_dot_a_local = valM
    grad_dot_lazy_fw_vertex = val

    if strong_lazification
        a_taken = a_local
        grad_dot_a_taken = grad_dot_a_local
        # in-face LMO is called directly
    else
        a_taken = compute_inface_extreme_point(lmo, NegatingArray(gradient), x)
        grad_dot_a_taken = fast_dot(gradient, a_taken)
    end

    # Do lazy pairwise step
    if grad_dot_a_taken - grad_dot_lazy_fw_vertex >= phi &&
       grad_dot_a_taken - grad_dot_lazy_fw_vertex >= epsilon
        step_type = ST_LAZY
        v = v_local
        a = a_taken
        d = muladd_memory_mode(memory_mode, d, a, v)
        fw_index = v_local_loc
    else
        v = compute_extreme_point(lmo, gradient)
        grad_dot_v = fast_dot(gradient, v)
        dual_gap = grad_dot_x - grad_dot_v

        if grad_dot_a_taken - grad_dot_v >= phi/lazy_tolerance &&
           grad_dot_a_taken - grad_dot_v >= epsilon
            a = a_taken
            d = muladd_memory_mode(memory_mode, d, a, v)
            step_type = strong_lazification ? ST_LAZY : ST_PAIRWISE
            away_index = strong_lazification ? a_local_loc : nothing
        elseif dual_gap >= phi / lazy_tolerance
            if strong_lazification
                a = compute_inface_extreme_point(lmo, NegatingArray(gradient), x)
            else
                a = a_taken
            end
            d = muladd_memory_mode(memory_mode, d, a, v)
            # lower our expectation
        else
            step_type = ST_DUALSTEP
            phi = min(dual_gap, phi / 2.0)
            a = a_taken
            d = zeros(length(x))
        end
    end

    return d, v, fw_index, a, away_index, phi, step_type
end

function lazy_blended_dicg_step(
    x,
    gradient,
    lmo,
    pre_computed_set,
    phi,
    epsilon,
    d;
    strong_lazification = false,
    lazy_tolerance = 2.0,
    memory_mode::MemoryEmphasis = InplaceEmphasis(),
)
    v_local, v_local_loc, val, a_local, a_local_loc, valM =
        pre_computed_set_argminmax(lmo, pre_computed_set, gradient, x; strong_lazification = strong_lazification)
    step_type = ST_PAIRWISE
    away_index = nothing
    fw_index = nothing
    grad_dot_x = fast_dot(x, gradient)
    grad_dot_a_local = valM
    grad_dot_lazy_fw_vertex = val

    if strong_lazification
        a_taken = a_local
        v_taken = v_local
        grad_dot_a_taken = grad_dot_a_local
        grad_dot_v_taken = grad_dot_lazy_fw_vertex
    else
        a_taken = compute_inface_extreme_point(lmo, NegatingArray(gradient), x)
        v_taken = compute_inface_extreme_point(lmo, gradient, x)
        grad_dot_a_taken = fast_dot(gradient, a_taken)
        grad_dot_v_taken = fast_dot(gradient, v_taken)
    end

    # Do lazy pairwise step
    if grad_dot_a_taken - grad_dot_v_taken >= phi &&
       grad_dot_a_taken - grad_dot_v_taken >= epsilon
        step_type = ST_LAZY
        v = v_taken
        a = a_taken
        d = muladd_memory_mode(memory_mode, d, a, v)
        fw_index = v_local_loc
        away_index = a_local_loc
    else
        if strong_lazification
            v_inface = compute_inface_extreme_point(lmo, gradient)
            grad_dot_v_inface = fast_dot(gradient, v_inface)

            if grad_dot_a_taken - grad_dot_v_inface >= phi &&
               grad_dot_a_taken - grad_dot_v_inface >= epsilon
                step_type = ST_LAZY
                v = v_inface
                a = a_taken
                d = muladd_memory_mode(memory_mode, d, a, v)
                away_index = a_local_loc
            end
        else
            v_inface = v_taken
            grad_dot_v_inface = grad_dot_v_taken
        end

        if step_type !== ST_LAZY
            v = compute_extreme_point(lmo, gradient)
            grad_dot_v = fast_dot(gradient, v)
            dual_gap = grad_dot_x - grad_dot_v
            if dual_gap >= phi / lazy_tolerance

                if strong_lazification
                    a_taken = compute_inface_extreme_point(lmo, NegatingArray(gradient), x)
                    grad_dot_a_taken = fast_dot(gradient, a_taken)
                end

                if grad_dot_a_taken - grad_dot_v_inface >= grad_dot_x - grad_dot_v / lazy_tolerance
                    step_type = ST_PAIRWISE
                    a = a_taken
                    d = muladd_memory_mode(memory_mode, d, a, v_inface)
                else
                    step_type = ST_REGULAR
                    a = x
                    d = muladd_memory_mode(memory_mode, d, x, v)
                end
            else
                step_type = ST_DUALSTEP
                phi = min(dual_gap, phi / 2.0)
                a = a_taken
                d = zeros(length(x))
            end
        end
    end
    return d, v, fw_index, a, away_index, phi, step_type
end
