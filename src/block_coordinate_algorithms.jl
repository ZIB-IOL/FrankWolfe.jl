"""
Update order for a block-coordinate method.
A `BlockCoordinateUpdateOrder` must implement
```
select_update_indices(::BlockCoordinateUpdateOrder, l)
```
"""
abstract type BlockCoordinateUpdateOrder end

"""
    select_update_indices(::BlockCoordinateUpdateOrder, l)

Returns a list of lists of the indices, where `l` is largest index i.e. the number of blocks. Each list represents one round of updates in an iteration. The indices in a list show which blocks should be updated parallely in one round.
For example, a full update is given by `[1:l]` and a blockwise update by `[[i] for i=1:l]`.
"""
function select_update_indices end

struct FullUpdate <: BlockCoordinateUpdateOrder end
struct CyclicUpdate <: BlockCoordinateUpdateOrder end
struct StochasticUpdate <: BlockCoordinateUpdateOrder end

function select_update_indices(::FullUpdate, l)
    return [1:l]
end

function select_update_indices(::CyclicUpdate, l)
    return [[i] for i in 1:l]
end

function select_update_indices(::StochasticUpdate, l)
    return [[rand(1:l)] for i in 1:l]
end

struct CallbackStateBlockCoordinateMethod{TP,TDV,TDG,XT,VT,TG,FT,GFT,LMO,GT}
    t::Int
    primal::TP
    dual::TDV
    dual_gap::TDG
    infeas::Float64
    time::Float64
    x::XT
    v::VT
    gamma::TG
    f::FT
    grad!::GFT
    lmo::LMO
    gradient::GT
    tt::FrankWolfe.StepType
end


function callback_state(state::CallbackStateBlockCoordinateMethod)
    return (state.t, state.primal, state.dual, state.dual_gap, state.time, state.infeas)
end

"""
    block_coordinate_frank_wolfe(f, grad!, lmo::ProductLMO, x0; ...)

Block-coordinate version of the vanilla Frank-Wolfe algorithm.
Minimizes objective `f` over product of feasible domains specified by the `lmo`.
The optional argument the `update_order` controls the order in which the blocks are updated (see [`FrankWolfe.BlockCoordinateUpdateOrder`](@ref)).

The method returns a tuple `(x, v, primal, dual_gap, infeas, traj_data)` with:
- `x` cartesian product of final iterates
- `v` cartesian product of last vertices of the LMOs
- `primal` primal value `f(x)`
- `dual_gap` final Frank-Wolfe gap
- `traj_data` vector of trajectory information.
"""
function block_coordinate_frank_wolfe(
    f,
    grad!,
    lmo::ProductLMO{N},
    x0;
    update_order::BlockCoordinateUpdateOrder=CyclicUpdate(),
    line_search::LineSearchMethod=Adaptive(),
    momentum=nothing,
    epsilon=1e-7,
    max_iteration=10000,
    print_iter=1000,
    trajectory=false,
    verbose=false,
    memory_mode=InplaceEmphasis(),
    gradient=nothing,
    callback=nothing,
    traj_data=[],
    timeout=Inf,
    linesearch_workspace=nothing,
) where {N}

    # header and format string for output of the algorithm
    headers = ["Type", "Iteration", "Primal", "Dual", "Dual Gap", "Infeas", "Time", "It/sec"]
    format_string = "%6s %13s %14e %14e %14e %14e %14e %14e\n"
    function format_state(state)
        rep = (
            st[Symbol(state.tt)],
            string(state.t),
            Float64(state.primal),
            Float64(state.primal - state.dual_gap),
            Float64(state.dual_gap),
            Float64(state.infeas),
            state.time,
            state.t / state.time,
        )
        return rep
    end

    ndim = ndims(x0)
    t = 0
    dual_gap = Inf
    dual_gaps = fill(Inf, N)
    primal = Inf
    x = copy(x0)
    tt = regular


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
        println("\nBlock coordinate Frank-Wolfe (BCFW).")
        num_type = eltype(x0[1])
        println(
            "MEMORY_MODE: $memory_mode STEPSIZE: $line_search EPSILON: $epsilon MAXITERATION: $max_iteration TYPE: $num_type",
        )
        grad_type = typeof(gradient)
        println("MOMENTUM: $momentum GRADIENTTYPE: $grad_type")
        if memory_mode isa InplaceEmphasis
            @info("In memory_mode memory iterates are written back into x0!")
        end
    end

    first_iter = true
    # instanciating container for gradient
    if gradient === nothing
        gradient = similar(x)
    end
    if linesearch_workspace === nothing
        linesearch_workspace = build_linesearch_workspace(line_search, x, gradient) # TODO: might not be really needed - hence hack
    end

    # container for direction
    d = similar(x)
    gtemp = if momentum === nothing
        d
    else
        similar(x)
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

        first_iter = false

        for update_indices in select_update_indices(update_order, N)

            # Update gradients
            if momentum === nothing || first_iter
                grad!(gradient, x)
                if momentum !== nothing
                    gtemp .= gradient
                end
            else
                grad!(gtemp, x)
                @memory_mode(memory_mode, gradient = (momentum * gradient) + (1 - momentum) * gtemp)
            end

            v = copy(x) # This is equivalent to setting the rest of d to zero

            for i in update_indices
                multi_index = [idx < ndim ? Colon() : i for idx in 1:ndim]
                v[multi_index...] = compute_extreme_point(lmo.lmos[i], gradient[multi_index...])
                dual_gaps[i] =
                    fast_dot(x[multi_index...], gradient[multi_index...]) -
                    fast_dot(v[multi_index...], gradient[multi_index...])
            end

            dual_gap = sum(dual_gaps)

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

            x = muladd_memory_mode(memory_mode, x, gamma, d)
        end

        # go easy on the memory - only compute if really needed
        if (
            (mod(t, print_iter) == 0 && verbose) ||
            callback !== nothing ||
            line_search isa Shortstep
        )
            infeas = sum(
                fast_dot(
                    selectdim(x, ndim, i) - selectdim(x, ndim, j),
                    selectdim(x, ndim, i) - selectdim(x, ndim, j),
                ) for i in 1:N for j in 1:i-1
            )
            primal = f(x)

        end


        t = t + 1
        if callback !== nothing
            state = CallbackStateBlockCoordinateMethod(
                t,
                primal,
                primal - dual_gap,
                dual_gap,
                infeas,
                tot_time,
                x,
                v,
                gamma,
                f,
                grad!,
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

    grad!(gradient, x)
    v = cat(
        compute_extreme_point(lmo, tuple([selectdim(gradient, ndim, i) for i in 1:N]...))...,
        dims=ndim,
    )
    infeas = sum(
        fast_dot(
            selectdim(x, ndim, i) - selectdim(x, ndim, j),
            selectdim(x, ndim, i) - selectdim(x, ndim, j),
        ) for i in 1:N for j in 1:i-1
    )
    primal = f(x)
    dual_gap = fast_dot(x - v, gradient)

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
        state = CallbackStateBlockCoordinateMethod(
            t,
            primal,
            primal - dual_gap,
            dual_gap,
            infeas,
            tot_time,
            x,
            v,
            gamma,
            f,
            grad!,
            lmo,
            gradient,
            tt,
        )
        callback(state)
    end

    return x, v, primal, dual_gap, traj_data
end
