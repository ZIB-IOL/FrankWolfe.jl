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

Returns a list of lists of the indices, where `l` is largest index i.e. the number of blocks.
Each sublist represents one round of updates in an iteration. The indices in a list show which blocks should be updated parallely in one round.
For example, a full update is given by `[1:l]` and a blockwise update by `[[i] for i=1:l]`.
"""
function select_update_indices end

"""
The full update initiates a parallel update of all blocks in one single round.
"""
struct FullUpdate <: BlockCoordinateUpdateOrder end

"""
The cyclic update initiates a sequence of update rounds.
In each round only one block is updated. The order of the blocks is determined by the given order of the LMOs.
"""
struct CyclicUpdate <: BlockCoordinateUpdateOrder end

"""
The stochastic update initiates a sequence of update rounds.
In each round only one block is updated. The order of the blocks is a random.
"""
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

abstract type UpdateStep end

struct FrankWolfeStep <: UpdateStep end
struct BPFGStep <: UpdateStep end

function update(
    ::FrankWolfeStep,
    x,
    lmo,
    f,
    gradient,
    grad!,
    t,
    line_search,
    linesearch_workspace,
    memory_mode,
)
    linesearch_workspace = build_linesearch_workspace(line_search, x, gradient)
    d = similar(x)
    v = compute_extreme_point(lmo, gradient)
    dual_gap = fast_dot(x, gradient) - fast_dot(v, gradient)

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

    return (x, dual_gap, v, d, gamma)
end

"""
    block_coordinate_frank_wolfe(f, grad!, lmo::ProductLMO{N}, x0; ...) where {N}

Block-coordinate version of the Frank-Wolfe algorithm.
Minimizes objective `f` over the product of feasible domains specified by the `lmo`.
The optional argument the `update_order` is of type [FrankWolfe.BlockCoordinateUpdateOrder](@ref) and controls the order in which the blocks are updated.

The method returns a tuple `(x, v, primal, dual_gap, traj_data)` with:
- `x` cartesian product of final iterates
- `v` cartesian product of last vertices of the LMOs
- `primal` primal value `f(x)`
- `dual_gap` final Frank-Wolfe gap
- `traj_data` vector of trajectory information.

See [ S. Lacoste-Julien, M. Jaggi, M. Schmidt, and P. Pletscher 2013](https://arxiv.org/abs/1207.4747)
and [A. Beck, E. Pauwels and S. Sabach 2015](https://arxiv.org/abs/1502.03716) for more details about Block-Coordinate Frank-Wolfe.
"""
function block_coordinate_frank_wolfe(
    f,
    grad!,
    lmo::ProductLMO{N},
    x0;
    update_order::BlockCoordinateUpdateOrder=CyclicUpdate(),
    line_search::LS=Adaptive(),
    update_step::US=FrankWolfeStep(),
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
) where {N,US<:Union{FrankWolfeStep,NTuple{N,FrankWolfeStep}}, LS<:Union{LineSearchMethod,NTuple{N,LineSearchMethod}}}

    # header and format string for output of the algorithm
    headers = ["Type", "Iteration", "Primal", "Dual", "Dual Gap", "Time", "It/sec"]
    format_string = "%6s %13s %14e %14e %14e %14e %14e\n"
    function format_state(state)
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

    if update_step isa FrankWolfeStep
        update_step = [update_step for _ in 1:N]
    end

    if line_search isa LineSearchMethod
        line_search = [line_search for _ in 1:N]
    end

    gamma = nothing
    v = similar(x)

    time_start = time_ns()

    if (momentum !== nothing && line_search isa Union{Shortstep,Adaptive,Backtracking})
        @warn("Momentum-averaged gradients should usually be used with agnostic stepsize rules.",)
    end

    # instanciating container for gradient
    if gradient === nothing
        gradient = collect(x)
    end

    if verbose
        println("\nBlock coordinate Frank-Wolfe (BCFW).")
        num_type = eltype(x0[1])
        line_search_type = [typeof(a) for a in line_search]
        println(
            "MEMORY_MODE: $memory_mode STEPSIZE: $line_search_type EPSILON: $epsilon MAXITERATION: $max_iteration TYPE: $num_type",
        )
        grad_type = typeof(gradient)
        println("MOMENTUM: $momentum GRADIENTTYPE: $grad_type UPDATE_ORDER: $update_order")
        if memory_mode isa InplaceEmphasis
            @info("In memory_mode memory iterates are written back into x0!")
        end
    end

    first_iter = true
    if linesearch_workspace === nothing
        linesearch_workspace = [build_linesearch_workspace(line_search[i], x, gradient) for i=1:N] # TODO: might not be really needed - hence hack
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

            xold = copy(x)
            for i in update_indices
                multi_index = [idx < ndim ? Colon() : i for idx in 1:ndim]
                pre_index = [idx < ndim ? Colon() : 1:i-1 for idx in 1:ndim]
                x_pre = xold[pre_index...]
                post_index = [idx < ndim ? Colon() : i+1:N for idx in 1:ndim]
                x_post = xold[post_index...]

                function temp_grad!(storage, y, i)
                    z = cat(x_pre, y, x_post; dims=ndim)
                    big_storage = similar(z)
                    grad!(big_storage, z),
                    @. storage = big_storage[i]
                end

                x_i, dual_gap_i, v, d, gamma = update(
                    update_step[i],
                    xold[multi_index...],
                    lmo.lmos[i],
                    y -> f(cat(x_pre, y, x_post; dims=ndim)),
                    gradient[multi_index...],
                    (storage, y) -> temp_grad!(storage, y, i),
                    t,
                    line_search[i],
                    linesearch_workspace[i],
                    memory_mode,
                )

                dual_gaps[i] = dual_gap_i
                x[multi_index...] = x_i
            end

            dual_gap = sum(dual_gaps)
        end

        # go easy on the memory - only compute if really needed
        if (
            (mod(t, print_iter) == 0 && verbose) ||
            callback !== nothing ||
            line_search isa Shortstep
        )
            primal = f(x)
        end


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

    primal = f(x)
    dual_gap = fast_dot(x - v, gradient)

    tot_time = (time_ns() - time_start) / 1.0e9

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
            tt,
        )
        callback(state)
    end

    return x, v, primal, dual_gap, traj_data
end
