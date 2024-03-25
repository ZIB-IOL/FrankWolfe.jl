
## interface functions for LMOs that are supported by the decomposition-invariant algorithm

"""
    is_decomposition_invariant_oracle(lmo)

Function to indicate whether the given LMO supports the decomposition-invariant interface.
This interface includes `compute_extreme_point` with a `lazy` keyword, `compute_inface_away_point`
and `dicg_maximum_step`.
"""
is_decomposition_invariant_oracle(::LinearMinimizationOracle) = false

"""
    compute_inface_away_point(lmo, direction, x; lazy, kwargs...)

LMO-like operation which computes a vertex maximizing in the `direction` on the face defined by the current fixings.
Fixings are maintained by the oracle (or deduced from `x` itself).
"""
function compute_inface_away_point(lmo, direction, x; lazy, kwargs...) end

"""
    dicg_maximum_step(lmo, x, direction)

Given `x` the current iterate and `direction` the negative of the direction towards which the iterate will move,
determine a maximum step size `gamma_max`, such that `x - gamma_max * direction` is in the polytope.
"""
function dicg_maximum_step(lmo, x, direction) end

struct ZeroOneHypercube
end

function compute_extreme_point(lmo::ZeroOneHypercube, direction; lazy=false, kwargs...)
    v = BitVector(signbit(di) for di in direction)
    return v
end

function compute_inface_away_point(lmo::ZeroOneHypercube, direction, x; lazy=false, kwargs...)
    v = BitVector(signbit(-di) for di in direction)
    for idx in eachindex(x)
        if x[idx] ≈ 1
            v[idx] = true
        end
        if x[idx] ≈ 0
            v[idx] = false
        end
    end
    return v
end

is_decomposition_invariant_oracle(::ZeroOneHypercube) = true

"""
Find the maximum step size γ such that `x - γ d` remains in the feasible set.
"""
function dicg_maximum_step(lmo::ZeroOneHypercube, x, direction)
    T = promote_type(eltype(x), eltype(direction))
    gamma_max = one(T)
    for idx in eachindex(x)
        if direction[idx] != 0.0
            # iterate already on the boundary
            if (direction[idx] < 0 && x[idx] ≈ 1) || (direction[idx] > 0 && x[idx] ≈ 0)
                return zero(gamma_max)
            end
            # clipping with the zero boundary
            if direction[idx] > 0
                gamma_max = min(gamma_max, x[idx] / direction[idx])
            else
                @assert direction[idx] < 0
                gamma_max = min(gamma_max, -(1 - x[idx]) / direction[idx])
            end
        end
    end
    return gamma_max
end

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
    renorm_interval=1000,
    lazy=false,
    linesearch_workspace=nothing,
    lazy_tolerance=2.0,
    recompute_last_vertex=true,
)

    if !is_decomposition_invariant_oracle(lmo)
        error("The provided LMO of type $(typeof(lmo)) does not support the decomposition-invariant interface")
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

    t = 0
    primal = convert(eltype(x0), Inf)
    tt = regular
    time_start = time_ns()

    x = x0
    d = similar(x)

    if gradient === nothing
        gradient = collect(x)
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
        if memory_mode isa InplaceEmphasis
            @info("In memory_mode memory iterates are written back into x0!")
        end
    end

    grad!(gradient, x)
    v = x0
    # if !lazy, phi is maintained as the global dual gap
    phi = max(0, fast_dot(x, gradient) - fast_dot(v, gradient))
    local_gap = zero(phi)
    gamma = one(phi)

    # active set used to store vertices
    # only relevant for lazification
    active_set = ActiveSet([1.0], x0)

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
            error("not implemented yet")
            # _, v_local, v_local_loc, _, a_lambda, a, a_loc, _, _ =
            #     active_set_argminmax(active_set, gradient)

            # dot_forward_vertex = fast_dot(gradient, v_local)
            # dot_away_vertex = fast_dot(gradient, a)
            # local_gap = dot_away_vertex - dot_forward_vertex
        else # non-lazy, call the simple and modified
            if t > 1
                v = compute_extreme_point(lmo, gradient, lazy=lazy)
                dual_gap = fast_dot(gradient, x) - fast_dot(gradient, v)
                phi = dual_gap
            end
            a = compute_inface_away_point(lmo, gradient, x; lazy=lazy)
        end
        d = muladd_memory_mode(memory_mode, d, a, v)
        gamma_max = dicg_maximum_step(lmo, x, d)
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
    end

    # recompute everything once more for final verfication / do not record to trajectory though
    # this is important as some variants do not recompute f(x) and the dual_gap regularly but only when reporting
    # hence the final computation.
    # do also cleanup of active_set due to many operations on the same set

    if verbose
        grad!(gradient, x)
        v = compute_extreme_point(lmo, gradient)
        primal = f(x)
        phi = fast_dot(x, gradient) - fast_dot(v, gradient)
        tt = last
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
                tt,
            )
            callback(state)
        end
    end
    return (x=x, v=v, primal=primal, dual_gap=dual_gap, traj_data=traj_data)
end
