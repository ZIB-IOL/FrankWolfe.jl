abstract type BlockCoordinateMethod end

function perform_bc_updates end

abstract type BlockCoordinateUpdateOrder end

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


mutable struct BCFW{MT,GT,CT,TT,LT} <: BlockCoordinateMethod
    update_order::BlockCoordinateUpdateOrder
    line_search::LineSearchMethod
    momentum::MT
    epsilon::Float64
    max_iteration::Any
    print_iter::Any
    trajectory::Bool
    verbose::Bool
    memory_mode::MemoryEmphasis
    gradient::GT
    callback::CT
    traj_data::TT
    timeout::Float64
    linesearch_workspace::LT
end

BCFW(;
    update_order=CyclicUpdate(),
    line_search=Adaptive(),
    momentum=nothing,
    epsilon=1e-7,
    max_iteration=10000,
    print_iter=1000.0,
    trajectory=false,
    verbose=false,
    memory_mode=InplaceEmphasis(),
    gradient=nothing,
    callback=nothing,
    traj_data=[],
    timeout=Inf,
    linesearch_workspace=nothing,
) = BCFW(
    update_order,
    line_search,
    momentum,
    epsilon,
    max_iteration,
    print_iter,
    trajectory,
    verbose,
    memory_mode,
    gradient,
    callback,
    traj_data,
    timeout,
    linesearch_workspace,
)

struct CallbackStateBCFW{TP,TDV,TDG,XT,VT,TG,FT,GFT,LMO,GT}
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


function callback_state(state::CallbackStateBCFW)
    return (state.t, state.primal, state.dual, state.dual_gap, state.time, state.infeas)
end

function perform_bc_updates(bc_algo::BCFW, f, grad!, lmo, x0)


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

    l = length(lmo.lmos)
    ndim = ndims(x0)
    t = 0
    dual_gap = Inf
    dual_gaps = fill(Inf, l)
    primal = Inf
    x = copy(x0)
    tt = regular

    line_search = bc_algo.line_search
    update_order = bc_algo.update_order
    verbose = bc_algo.verbose
    momentum = bc_algo.momentum
    epsilon = bc_algo.epsilon
    max_iteration = bc_algo.max_iteration
    print_iter = bc_algo.print_iter
    trajectory = bc_algo.trajectory
    verbose = bc_algo.verbose
    memory_mode = bc_algo.memory_mode
    gradient = bc_algo.gradient
    callback = bc_algo.callback
    traj_data = bc_algo.traj_data
    timeout = bc_algo.timeout
    linesearch_workspace = bc_algo.linesearch_workspace


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

        for update_indices in select_update_indices(update_order, l)
            
            # Update gradients
            if momentum === nothing || first_iter
                grad!(gradient, x)
                if momentum !== nothing
                    gtemp .= gradient
                end
            else
                grad!(gtemp, x)
                @memory_mode(
                    memory_mode,
                    gradient = (momentum * gradient) + (1 - momentum) * gtemp
                )
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
                ) for i in 1:l for j in 1:i-1
            )
            primal = f(x)

        end


        t = t + 1
        if callback !== nothing
            state = CallbackStateBCFW(
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
        compute_extreme_point(lmo, tuple([selectdim(gradient, ndim, i) for i in 1:l]...))...,
        dims=ndim,
    )
    infeas = sum(
        fast_dot(
            selectdim(x, ndim, i) - selectdim(x, ndim, j),
            selectdim(x, ndim, i) - selectdim(x, ndim, j),
        ) for i in 1:l for j in 1:i-1
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
        state = CallbackStateBCFW(
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

    return x, v, primal, dual_gap, infeas, traj_data
end
