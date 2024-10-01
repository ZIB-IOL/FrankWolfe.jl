"""
Update order for a block-coordinate method.
A `BlockCoordinateUpdateOrder` must implement
```
select_update_indices(::BlockCoordinateUpdateOrder, s::CallbackState, dual_gaps)
```
"""
abstract type BlockCoordinateUpdateOrder end

"""
    select_update_indices(::BlockCoordinateUpdateOrder, s::CallbackState, dual_gaps)

Returns a list of lists of the block indices.
Each sublist represents one round of updates in an iteration. The indices in a list show which blocks should be updated parallely in one round.
For example, a full update is given by `[1:l]` and a blockwise update by `[[i] for i=1:l]`, where `l` is the number of blocks.
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
struct CyclicUpdate <: BlockCoordinateUpdateOrder
    limit::Int
end

CyclicUpdate() = CyclicUpdate(-1)

"""
The stochastic update initiates a sequence of update rounds.
In each round only one block is updated. The order of the blocks is a random.
"""
struct StochasticUpdate <: BlockCoordinateUpdateOrder
    limit::Int
end

StochasticUpdate() = StochasticUpdate(-1)

mutable struct DualGapOrder <: BlockCoordinateUpdateOrder
    limit::Int
end

DualGapOrder() = DualGapOrder(-1)

mutable struct DualProgressOrder <: BlockCoordinateUpdateOrder
    limit::Int
    previous_dual_gaps::Vector{Float64}
    dual_progress::Vector{Float64}
end

DualProgressOrder() = DualProgressOrder(-1, [], [])
DualProgressOrder(i::Int64) = DualProgressOrder(i, [], [])

"""
The Lazy update order is discussed in "Flexible block-iterative
analysis for the Frank-Wolfe algorithm," by Braun, Pokutta, &
Woodstock (2024). 
'lazy_block' is an index of a computationally expensive block to
update;
'refresh_rate' describes the frequency at which we perform
a full activation; and
'block_size' describes the number of "faster" blocks
(i.e., those excluding 'lazy_block') activated (chosen
uniformly at random) during each
of the "faster" iterations; for more detail, see the
article. If  'block_size' is unspecified, this defaults to
1. 
Note: This methodology is currently only proven to work
with 'FrankWolfe.Shortstep' linesearches and a (not-yet
implemented) adaptive method; see the article for details.
"""
struct LazyUpdate <: BlockCoordinateUpdateOrder 
    lazy_block::Int
    refresh_rate::Int
    block_size::Int
end

function LazyUpdate(lazy_block::Int,refresh_rate::Int)
    return LazyUpdate(lazy_block, refresh_rate, 1)
end

function select_update_indices(::FullUpdate, s::CallbackState, _)
    return [1:length(s.lmo.lmos)]
end

function select_update_indices(u::CyclicUpdate, s::CallbackState, _)

    l = length(s.lmo.lmos)

    @assert u.limit <= l
    @assert u.limit > 0 || u.limit == -1

    if u.limit in [-1, l]
        return [[i] for i in 1:l]
    end 

    start = (s.t*u.limit % l) + 1
    if start + u.limit - 1 ≤ l
        return [[i] for i in start:start + u.limit - 1]
    else
        a = [[i] for i in start:l]
        append!(a, [[i] for i in 1:u.limit-length(a)])
        return a
    end
end

function select_update_indices(u::StochasticUpdate, s::CallbackState, _)

    l = length(s.lmo.lmos)

    @assert u.limit <= l
    @assert u.limit > 0 || u.limit == -1
    
    if u.limit == -1
        return [[rand(1:l)] for i in 1:l]
    end
    return [[rand(1:l)] for i in 1:u.limit]
end

function select_update_indices(u::DualProgressOrder, s::CallbackState, dual_gaps)

    l = length(s.lmo.lmos)

    @assert u.limit <= l
    @assert u.limit > 0 || u.limit == -1

    # In the first iteration update every block so we get finite dual progress
    if s.t < 2 
        u.previous_dual_gaps = copy(dual_gaps)
        return [[i] for i=1:l] 
    end

    if s.t == 1
        u.dual_progress = dual_gaps - u.previous_dual_gaps
    else
        diff = dual_gaps - u.previous_dual_gaps
        u.dual_progress = [d == 0 ? u.dual_progress[i] : d for (i, d) in enumerate(diff)]
    end
    u.previous_dual_gaps = copy(dual_gaps)

    n = u.limit == -1 ? l : u.limit
    return [[findfirst(cumsum(u.dual_progress/sum(u.dual_progress)) .> rand())] for _=1:n]
end

function select_update_indices(u::DualGapOrder, s::CallbackState, dual_gaps)

    l = length(s.lmo.lmos)

    @assert u.limit <= l
    @assert u.limit > 0 || u.limit == -1

    # In the first iteration update every block so we get finite dual gaps
    if s.t < 1
        return [[i] for i=1:l] 
    end

    n = u.limit == -1 ? l : u.limit
    return [[findfirst(cumsum(dual_gaps/sum(dual_gaps)) .> rand())] for _=1:n]
end

function select_update_indices(update::LazyUpdate, s::CallbackState, dual_gaps)
    #Returns a sequence of randomized cheap indices by 
    #excluding update.lazy_block until "refresh_rate" updates
    #occur, then adds an update of everything while mainting
    #randomized order.
    l = length(s.lmo.lmos)
    return push!([[rand(range(1,l)[1:l .!= update.lazy_block]) for _ in range(1,update.block_size)] for _ in 1:(update.refresh_rate -1)], range(1,l))
end

"""
Update step for block-coordinate Frank-Wolfe.
These are implementations of different FW-algorithms to be used in a blockwise manner.
Each update step must implement
```
update_iterate(
    step::UpdateStep,
    x,
    lmo,
    f,
    gradient,
    grad!,
    dual_gap,
    t,
    line_search,
    linesearch_workspace,
    memory_mode,
    epsilon,
)
```
"""
abstract type UpdateStep end

"""
    update_iterate(
        step::UpdateStep,
        x,
        lmo,
        f,
        gradient,
        grad!,
        dual_gap,
        t,
        line_search,
        linesearch_workspace,
        memory_mode,
        epsilon,
    )
    
Executes one iteration of the defined [`FrankWolfe.UpdateStep`](@ref) and updates the iterate `x` implicitly.
The function returns a tuple `(dual_gap, v, d, gamma, step_type)`:
- `dual_gap` is the updated FrankWolfe gap
- `v` is the used vertex
- `d` is the update direction
- `gamma` is the applied step-size
- `step_type` is the applied step-type
"""
function update_iterate end

"""
Implementation of the vanilla Frank-Wolfe algorithm as an update step for block-coordinate Frank-Wolfe.
"""
struct FrankWolfeStep <: UpdateStep end

"""
Implementation of the blended pairwise conditional gradient (BPCG) method as an update step for block-coordinate Frank-Wolfe.
"""
mutable struct BPCGStep <: UpdateStep
    lazy::Bool
    active_set::Union{FrankWolfe.AbstractActiveSet,Nothing}
    renorm_interval::Int
    lazy_tolerance::Float64
    phi::Float64
end

function Base.copy(::FrankWolfeStep)
    return FrankWolfeStep()
end

function Base.copy(obj::BPCGStep)
    if obj.active_set === nothing
        return BPCGStep(false, nothing, obj.renorm_interval, obj.lazy_tolerance, Inf)
    else
        return BPCGStep(obj.lazy, copy(obj.active_set), obj.renorm_interval, obj.lazy_tolerance, obj.phi)
    end
end

BPCGStep() = BPCGStep(false, nothing, 1000, 2.0, Inf)

function update_iterate(
    ::FrankWolfeStep,
    x,
    lmo,
    f,
    gradient,
    grad!,
    dual_gap,
    t,
    line_search,
    linesearch_workspace,
    memory_mode,
    epsilon,
)
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

    step_type = ST_REGULAR

    return (dual_gap, v, d, gamma, step_type)
end

function update_iterate(
    s::BPCGStep,
    x,
    lmo,
    f,
    gradient,
    grad!,
    dual_gap,
    t,
    line_search,
    linesearch_workspace,
    memory_mode,
    epsilon,
)

    d = zero(x)
    step_type = ST_REGULAR

    _, v_local, v_local_loc, _, a_lambda, a, a_loc, _, _ =
        active_set_argminmax(s.active_set, gradient)

    dot_forward_vertex = fast_dot(gradient, v_local)
    dot_away_vertex = fast_dot(gradient, a)
    local_gap = dot_away_vertex - dot_forward_vertex

    if !s.lazy
        v = compute_extreme_point(lmo, gradient)
        dual_gap = fast_dot(gradient, x) - fast_dot(gradient, v)
        s.phi = dual_gap
    end

    # minor modification from original paper for improved sparsity
    # (proof follows with minor modification when estimating the step)
    if local_gap > s.phi / s.lazy_tolerance
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

        # reached maximum of lambda -> dropping away vertex
        if gamma ≈ gamma_max
            s.active_set.weights[v_local_loc] += gamma
            deleteat!(s.active_set, a_loc)
        else # transfer weight from away to local FW
            s.active_set.weights[a_loc] -= gamma
            s.active_set.weights[v_local_loc] += gamma
            @assert active_set_validate(s.active_set)
        end
        active_set_update_iterate_pairwise!(s.active_set.x, gamma, v_local, a)
    else # add to active set
        if s.lazy
            v = compute_extreme_point(lmo, gradient)
            step_type = ST_REGULAR
        end
        vertex_taken = v
        dual_gap = fast_dot(gradient, x) - fast_dot(gradient, v)
        # if we are about to exit, compute dual_gap with the cleaned-up x
        if dual_gap ≤ epsilon
            active_set_renormalize!(s.active_set)
            active_set_cleanup!(s.active_set)
            compute_active_set_iterate!(s.active_set)
            x = get_active_set_iterate(s.active_set)
            grad!(gradient, x)
            dual_gap = fast_dot(gradient, x) - fast_dot(gradient, v)
        end

        if !s.lazy || dual_gap ≥ s.phi / s.lazy_tolerance

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

            # dropping active set and restarting from singleton
            if gamma ≈ 1.0
                active_set_initialize!(s.active_set, v)
            else
                renorm = mod(t, s.renorm_interval) == 0
                active_set_update!(s.active_set, gamma, v, renorm, nothing)
            end
        else # dual step
            if step_type != ST_LAZYSTORAGE
                s.phi = dual_gap
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
            gamma = 0.0
        end
    end
    if mod(t, s.renorm_interval) == 0
        active_set_renormalize!(s.active_set)
    end

    x = muladd_memory_mode(memory_mode, x, gamma, d)

    return (dual_gap, vertex_taken, d, gamma, step_type)
end

"""
    block_coordinate_frank_wolfe(f, grad!, lmo::ProductLMO{N}, x0; ...) where {N}

Block-coordinate version of the Frank-Wolfe algorithm.
Minimizes objective `f` over the product of feasible domains specified by the `lmo`.
The optional argument the `update_order` is of type [`FrankWolfe.BlockCoordinateUpdateOrder`](@ref) and controls the order in which the blocks are updated.
The argument `update_step` is a single instance or tuple of [`FrankWolfe.UpdateStep`](@ref) and defines which FW-algorithms to use to update the iterates in the different blocks.

The method returns a tuple `(x, v, primal, dual_gap, traj_data)` with:
- `x` cartesian product of final iterates
- `v` cartesian product of last vertices of the LMOs
- `primal` primal value `f(x)`
- `dual_gap` final Frank-Wolfe gap
- `traj_data` vector of trajectory information.

See [S. Lacoste-Julien, M. Jaggi, M. Schmidt, and P. Pletscher 2013](https://arxiv.org/abs/1207.4747)
and [A. Beck, E. Pauwels and S. Sabach 2015](https://arxiv.org/abs/1502.03716) for more details about Block-Coordinate Frank-Wolfe.
"""
function block_coordinate_frank_wolfe(
    f,
    grad!,
    lmo::ProductLMO{N},
    x0::BlockVector;
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
) where {
    N,
    US<:Union{UpdateStep,NTuple{N,UpdateStep}},
    LS<:Union{LineSearchMethod,NTuple{N,LineSearchMethod}},
}

    # header and format string for output of the algorithm
    headers = ["Type", "Iteration", "Primal", "Dual", "Dual Gap", "Time", "It/sec"]
    format_string = "%6s %13s %14e %14e %14e %14e %14e\n"
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

    #ndim = ndims(x0)
    t = 0
    dual_gap = Inf
    dual_gaps = fill(Inf, N)
    primal = Inf
    x = copy(x0)
    step_type = ST_REGULAR

    if trajectory
        callback = make_trajectory_callback(callback, traj_data)
    end

    if verbose
        callback = make_print_callback(callback, print_iter, headers, format_string, format_state)
    end

    if update_step isa UpdateStep
        update_step = [copy(update_step) for _ in 1:N]
    end

    for (i, s) in enumerate(update_step)
        if s isa BPCGStep && s.active_set === nothing
            s.active_set = ActiveSet([(1.0, copy(x0.blocks[i]))])
        end
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
        gradient = similar(x)
    end

    if verbose
        println("\nBlock coordinate Frank-Wolfe (BCFW).")
        num_type = eltype(x0[1])
        line_search_type = [typeof(a) for a in line_search]
        println(
            "MEMORY_MODE: $memory_mode STEPSIZE: $line_search_type EPSILON: $epsilon MAXITERATION: $max_iteration TYPE: $num_type",
        )
        grad_type = typeof(gradient)
        update_step_type = [typeof(s) for s in update_step]
        println(
            "MOMENTUM: $momentum GRADIENTTYPE: $grad_type UPDATE_ORDER: $update_order UPDATE_STEP: $update_step_type",
        )
        if memory_mode isa InplaceEmphasis
            @info("In memory_mode memory iterates are written back into x0!")
        end
    end

    first_iter = true
    if linesearch_workspace === nothing
        linesearch_workspace = [
            build_linesearch_workspace(line_search[i], x.blocks[i], gradient.blocks[i]) for i in 1:N
        ] # TODO: might not be really needed - hence hack
    end

    # container for direction
    d = similar(x)
    gtemp = if momentum === nothing
        d
    else
        similar(x)
    end

    # container for state
    state = CallbackState(
        t,
        primal,
        primal - dual_gap,
        dual_gap,
        0.0,
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

        for update_indices in select_update_indices(update_order, state, dual_gaps)

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

            first_iter = false

            xold = copy(x)
            Threads.@threads for i in update_indices

                function extend(y)
                    bv = copy(xold)
                    bv.blocks[i] = y
                    return bv
                end

                function temp_grad!(storage, y, i)
                    z = extend(y)
                    big_storage = similar(z)
                    grad!(big_storage, z)
                    @. storage = big_storage.blocks[i]
                end

                dual_gaps[i], v.blocks[i], d.blocks[i], gamma, step_type = update_iterate(
                    update_step[i],
                    x.blocks[i],
                    lmo.lmos[i],
                    y -> f(extend(y)),
                    gradient.blocks[i],
                    (storage, y) -> temp_grad!(storage, y, i),
                    dual_gaps[i],
                    t,
                    line_search[i],
                    linesearch_workspace[i],
                    memory_mode,
                    epsilon,
                )
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
        if callback !== nothing || update_order isa CyclicUpdate
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
            if callback !== nothing
                # @show state
                if callback(state, dual_gaps) === false
                    break
                end
            end
        end

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
        callback(state, dual_gaps)
    end

    return x, v, primal, dual_gap, traj_data
end
