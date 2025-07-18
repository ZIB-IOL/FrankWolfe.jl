
"""
    momentum_iterate(iter::MomentumIterator) -> ρ

Method to implement for a type `MomentumIterator`.
Returns the next momentum value `ρ` and updates the iterator internal state.
"""
function momentum_iterate end

"""
    ExpMomentumIterator{T}

Iterator for the momentum used in the variant of Stochastic Frank-Wolfe.
Momentum coefficients are the values of the iterator:
`ρ_t = 1 - num / (offset + t)^exp`

The state corresponds to the iteration count.

Source:
Stochastic Conditional Gradient Methods: From Convex Minimization to Submodular Maximization
Aryan Mokhtari, Hamed Hassani, Amin Karbasi, JMLR 2020.
"""
mutable struct ExpMomentumIterator{T}
    exp::T
    num::T
    offset::T
    iter::Int
end

ExpMomentumIterator() = ExpMomentumIterator(2 / 3, 4.0, 8.0, 0)

function momentum_iterate(em::ExpMomentumIterator)
    em.iter += 1
    return 1 - em.num / (em.offset + em.iter)^(em.exp)
end

mutable struct InverseIterateMomentum{T}
    iter::Int
    factor::T
end

InverseIterateMomentum() = InverseIterateMomentum(0, true)

function momentum_iterate(it::InverseIterateMomentum)
    it.iter += 1
    return 1 - inv(it.iter^it.factor)
end

"""
    ConstantMomentumIterator{T}

Iterator for momentum with a fixed damping value, always return the value and a dummy state.
"""
struct ConstantMomentumIterator{T}
    v::T
end

momentum_iterate(em::ConstantMomentumIterator) = em.v

# batch sizes

"""
    batchsize_iterate(iter::BatchSizeIterator) -> b

Method to implement for a batch size iterator of type `BatchSizeIterator`.
Calling `batchsize_iterate` returns the next batch size and typically update the internal state of `iter`.
"""
function batchsize_iterate end

"""
    ConstantBatchIterator(batch_size)

Batch iterator always returning a constant batch size.
"""
struct ConstantBatchIterator
    batch_size::Int
end

batchsize_iterate(cbi::ConstantBatchIterator) = cbi.batch_size

"""
    IncrementBatchIterator(starting_batch_size, max_batch_size, [increment = 1])

Batch size starting at starting_batch_size and incrementing by `increment` at every iteration.
"""
mutable struct IncrementBatchIterator
    starting_batch_size::Int
    max_batch_size::Int
    increment::Int
    iter::Int
    maxreached::Bool
end

function IncrementBatchIterator(starting_batch_size::Int, max_batch_size::Int, increment::Int)
    return IncrementBatchIterator(starting_batch_size, max_batch_size, increment, 0, false)
end

function IncrementBatchIterator(starting_batch_size::Int, max_batch_size::Int)
    return IncrementBatchIterator(starting_batch_size, max_batch_size, 1, 0, false)
end

function batchsize_iterate(ibi::IncrementBatchIterator)
    if ibi.maxreached
        return ibi.max_batch_size
    end
    new_size = ibi.starting_batch_size + ibi.iter * ibi.increment
    ibi.iter += 1
    if new_size > ibi.max_batch_size
        ibi.maxreached = true
        return ibi.max_batch_size
    end
    return new_size
end

"""
    stochastic_frank_wolfe(f::StochasticObjective, lmo, x0; kwargs...)

Stochastic version of Frank-Wolfe, evaluates the objective and gradient stochastically,
implemented through the [`FrankWolfe.StochasticObjective`](@ref) interface.

$COMMON_ARGS

$COMMON_KWARGS

# Specific keyword arguments

Keyword arguments include `batch_size` to pass a fixed `batch_size`
or a `batch_iterator` implementing
`batch_size = FrankWolfe.batchsize_iterate(batch_iterator)` for algorithms like
Variance-reduced and projection-free stochastic optimization, E Hazan, H Luo, 2016.

Similarly, a constant `momentum` can be passed or replaced by a `momentum_iterator`
implementing `momentum = FrankWolfe.momentum_iterate(momentum_iterator)`.

The keyword `use_full_evaluation` set to true allows the algorithm to compute the deterministic primal value and FW gap.

The One-Sample Stochastic Frank-Wolfe (1SFW) can be activated with `use_one_sample_variant`.
$RETURN
"""
function stochastic_frank_wolfe(
    f::AbstractStochasticObjective,
    lmo,
    x0;
    line_search::LineSearchMethod=Nonconvex(),
    momentum_iterator=nothing,
    momentum=nothing,
    epsilon=1e-7,
    max_iteration=10000,
    print_iter=1000,
    trajectory=false,
    verbose=false,
    memory_mode::MemoryEmphasis=InplaceEmphasis(),
    rng=Random.GLOBAL_RNG,
    batch_size=length(f.xs) ÷ 10 + 1,
    batch_iterator=nothing,
    use_full_evaluation=false,
    callback=nothing,
    traj_data=[],
    timeout=Inf,
    linesearch_workspace=nothing,
    use_one_sample_variant=false,
)

    # format string for output of the algorithm
    format_string = "%6s %13s %14e %14e %14e %14e %14e %6i\n"
    headers = ("Type", "Iteration", "Primal", "Dual", "Dual Gap", "Time", "It/sec", "Batch")

    function format_state(state, batch_size)
        rep = (
            steptype_string[Symbol(state.step_type)],
            string(state.t),
            Float64(state.primal),
            Float64(state.primal - state.dual_gap),
            Float64(state.dual_gap),
            state.time,
            state.t / state.time,
            batch_size,
        )
        return rep
    end

    t = 0
    dual_gap = Inf
    primal = Inf
    v = []
    x = x0
    d = similar(x)
    step_type = ST_REGULAR

    if trajectory
        callback = make_trajectory_callback(callback, traj_data)
    end

    if verbose
        callback = make_print_callback(callback, print_iter, headers, format_string, format_state)
    end

    time_start = time_ns()

    if momentum_iterator === nothing && momentum !== nothing
        momentum_iterator = ConstantMomentumIterator(momentum)
    end
    if batch_iterator === nothing
        batch_iterator = ConstantBatchIterator(batch_size)
    end

    if verbose
        println(
            "\n" * use_one_sample_variant ? "One-sample " : "" *
            "Stochastic Frank-Wolfe Algorithm."
        )
        NumType = eltype(x0)
        println(
            "MEMORY_MODE: $memory_mode STEPSIZE: $line_search EPSILON: $epsilon max_iteration: $max_iteration TYPE: $NumType",
        )
        println(
            "GRADIENTTYPE: $(typeof(f.storage)) MOMENTUM: $(momentum_iterator !== nothing) BATCH_POLICY: $(typeof(batch_iterator)) ",
        )
        println("LMO: $(typeof(lmo))")
        if memory_mode isa InplaceEmphasis
            @info("In memory_mode memory iterates are written back into x0!")
        end
    end

    if memory_mode isa InplaceEmphasis && !isa(x, Union{Array,SparseArrays.AbstractSparseArray})
        if eltype(x) <: Integer
            x = copyto!(similar(x, float(eltype(x))), x)
        else
            x = copyto!(similar(x), x)
        end
    end
    first_iter = true
    gradient = f.storage .* 0
    if use_one_sample_variant
        previous_gradient = 0 .* gradient
    end
    if linesearch_workspace === nothing
        linesearch_workspace = build_linesearch_workspace(line_search, x, gradient)
    end

    while t <= max_iteration

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
        batch_size = batchsize_iterate(batch_iterator)

        if momentum_iterator === nothing
            gradient = compute_gradient(f, x, rng=rng, batch_size=batch_size)
        elseif first_iter
            compute_gradient(f, x, rng=rng, batch_size=batch_size)
            gradient .= f.storage
        else
            momentum = momentum_iterate(momentum_iterator)
            if !use_one_sample_variant
                compute_gradient(f, x, rng=rng, batch_size=batch_size, full_evaluation=false)
                # gradient = momentum * gradient + (1 - momentum) * f.storage
                LinearAlgebra.mul!(gradient, LinearAlgebra.I, f.storage, 1 - momentum, momentum)
            else
                copyto!(previous_gradient, f.storage)
                compute_gradient(f, x, rng=rng, batch_size=batch_size, full_evaluation=false)
                # gradient = momentum * (gradient - prev_gradient) + f.storage
                LinearAlgebra.mul!(gradient, LinearAlgebra.I, previous_gradient, -momentum, momentum)
                LinearAlgebra.mul!(gradient, LinearAlgebra.I, f.storage, 1, 1)
            end
        end
        first_iter = false

        v = compute_extreme_point(lmo, gradient)

        # go easy on the memory - only compute if really needed
        compute_iter =
            (mod(t, print_iter) == 0 && verbose) ||
            callback !== nothing ||
            !(line_search isa Agnostic || line_search isa Nonconvex || line_search isa FixedStep)
        if compute_iter
            primal = compute_value(f, x, full_evaluation=use_full_evaluation)
            dual_gap = dot(gradient, x) - dot(gradient, v)
        end

        d = muladd_memory_mode(memory_mode, d, x, v)

        # note: only agnostic line-search methods are supported
        # so nothing is passed as function and gradient
        gamma = perform_line_search(
            line_search,
            t,
            nothing,
            nothing,
            gradient,
            x,
            d,
            1.0,
            linesearch_workspace,
            memory_mode,
        )
        t += 1
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
                nothing,
                lmo,
                gradient,
                step_type,
            )
            if callback(state, batch_size) === false
                break
            end
        end

        x = muladd_memory_mode(memory_mode, x, gamma, d)
    end
    # recompute everything once for final verfication / no additional callback call
    # this is important as some variants do not recompute f(x) and the dual_gap regularly but only when reporting
    # hence the final computation.
    # last computation done with full evaluation if possible for exact gradient

    (primal, gradient) = compute_value_gradient(f, x, full_evaluation=use_full_evaluation)
    v = compute_extreme_point(lmo, gradient)
    dual_gap = dot(gradient, x) - dot(gradient, v)
    step_type = ST_LAST
    d = muladd_memory_mode(memory_mode, d, x, v)
    gamma = perform_line_search(
        line_search,
        t,
        nothing,
        nothing,
        gradient,
        x,
        d,
        1.0,
        linesearch_workspace,
        memory_mode,
    )
    tot_time = (time_ns() - time_start) / 1e9
    if callback !== nothing
        state = CallbackState(
            t,
            primal,
            primal - dual_gap,
            dual_gap,
            (time_ns() - time_start) / 1e9,
            x,
            v,
            d,
            gamma,
            f,
            nothing,
            lmo,
            gradient,
            step_type,
        )
        callback(state, batch_size)
    end
    return (x=x, v=v, primal=primal, dual_gap=dual_gap, traj_data=traj_data)
end
