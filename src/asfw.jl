using LinearAlgebra
using FrankWolfe

import MathOptInterface
const MOI = MathOptInterface
using GLPK
using SparseArrays

"""
    frank_wolfe(f, grad!, lmo, x0; ...)

Simplest form of the Frank-Wolfe algorithm.
Returns a tuple `(x, v, primal, dual_gap, traj_data)` with:
- `x` final iterate
- `v` last vertex from the LMO
- `primal` primal value `f(x)`
- `dual_gap` final Frank-Wolfe gap
- `traj_data` vector of trajectory information.
"""
function as_frank_wolfe(
    f,
    grad!,
    lmo,
    x0;
    line_search::FrankWolfe.LineSearchMethod=FrankWolfe.Adaptive(),
    momentum=nothing,
    epsilon=1e-7,
    max_iteration=10000,
    print_iter=1000,
    trajectory=false,
    verbose=false,
    memory_mode::FrankWolfe.MemoryEmphasis=FrankWolfe.InplaceEmphasis(),
    gradient=nothing,
    callback=nothing,
    traj_data=[],
    timeout=Inf,
    linesearch_workspace=nothing,
    dual_gap_compute_frequency=1,
)

    # header and format string for output of the algorithm
    headers = ["Type", "Iteration", "Primal", "Dual", "Dual Gap", "Time", "It/sec"]
    format_string = "%6s %13s %14e %14e %14e %14e %14e\n"
    function format_state(state)
        rep = (
            FrankWolfe.st[Symbol(state.tt)],
            string(state.t),
            Float64(state.primal),
            Float64(state.primal - state.dual_gap),
            Float64(state.dual_gap),
            state.time,
            state.t / state.time,
        )
        return rep
    end

    t = 0
    dual_gap = Inf
    primal = Inf
    v = []
    x = x0
    tt = FrankWolfe.regular

    if trajectory
        callback = FrankWolfe.make_trajectory_callback(callback, traj_data)
    end

    if verbose
        callback = FrankWolfe.make_print_callback(callback, print_iter, headers, format_string, format_state)
    end

    time_start = time_ns()

    if (momentum !== nothing && line_search isa Union{FrankWolfe.Shortstep, FrankWolfe.Adaptive, FrankWolfe.Backtracking})
        @warn("Momentum-averaged gradients should usually be used with agnostic stepsize rules.",)
    end

    if verbose
        println("\nVanilla Frank-Wolfe Algorithm.")
        NumType = eltype(x0)
        println(
            "MEMORY_MODE: $memory_mode STEPSIZE: $line_search EPSILON: $epsilon MAXITERATION: $max_iteration TYPE: $NumType",
        )
        grad_type = typeof(gradient)
        println("MOMENTUM: $momentum GRADIENTTYPE: $grad_type")
        println("LMO: $(typeof(lmo))")
        if memory_mode isa FrankWolfe.InplaceEmphasis
            @info("In memory_mode memory iterates are written back into x0!")
        end
    end
    if memory_mode isa FrankWolfe.InplaceEmphasis && !isa(x, Union{Array,SparseArrays.AbstractSparseArray})
        # if integer, convert element type to most appropriate float
        if eltype(x) <: Integer
            x = copyto!(similar(x, float(eltype(x))), x)
        else
            x = copyto!(similar(x), x)
        end
    end

    # instanciating container for gradient
    if gradient === nothing
        gradient = collect(x)
    end
    @show gradient # storage

    first_iter = true
    if linesearch_workspace === nothing
        linesearch_workspace = FrankWolfe.build_linesearch_workspace(line_search, x, gradient)
    end

    # container for direction
    d = similar(x)
    gtemp = momentum === nothing ? d : similar(x)

    while t <= max_iteration #&& dual_gap >= max(epsilon, eps(float(typeof(dual_gap))))

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


        if momentum === nothing || first_iter
            grad!(gradient, x)
            if momentum !== nothing
                gtemp .= gradient
            end
        else
            grad!(gtemp, x)
            FrankWolfe.@memory_mode(memory_mode, gradient = (momentum * gradient) + (1 - momentum) * gtemp)
        end

        v = if first_iter
            compute_extreme_point(lmo, gradient)
        else
            compute_extreme_point(lmo, gradient, v=v)
        end

        first_iter = false
        # go easy on runtime - only compute primal and dual if needed
        compute_iter = (
            (mod(t, print_iter) == 0 && verbose) ||
            callback !== nothing ||
            line_search isa FrankWolfe.Shortstep
        )
        if compute_iter
            primal = f(x)
        end
        # if t %  dual_gap_compute_frequency == 0 || compute_iter
        #     dual_gap = FrankWolfe.fast_dot(x, gradient) - FrankWolfe.fast_dot(v, gradient)
        # end
        d = FrankWolfe.muladd_memory_mode(memory_mode, d, x, v)

        gamma = FrankWolfe.perform_line_search(
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
        t = t + 1
        if callback !== nothing
            state = FrankWolfe.CallbackState(
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
            if callback(state) === false
                break
            end
        end

        x = FrankWolfe.muladd_memory_mode(memory_mode, x, gamma, d)
    end
    # recompute everything once for final verfication / do not record to trajectory though for now!
    # this is important as some variants do not recompute f(x) and the dual_gap regularly but only when reporting
    # hence the final computation.
    tt = last
    grad!(gradient, x)
    v = compute_extreme_point(lmo, gradient, v=v)
    primal = f(x)
    # dual_gap = FrankWolfe.fast_dot(x, gradient) - FrankWolfe.fast_dot(v, gradient)
    tot_time = (time_ns() - time_start) / 1.0e9
    gamma = FrankWolfe.perform_line_search(
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
        state = FrankWolfe.CallbackState(
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

mutable struct AbsSmoothLMO <: FrankWolfe.LinearMinimizationOracle
    fw_iteration_counter::Int
end

AbsSmoothLMO() = AbsSmoothLMO(0)

function FrankWolfe.compute_extreme_point(lmo::AbsSmoothLMO, direction; kwargs...)
    lmo.fw_iteration_counter += 1
    inner_iteration = 0
    t = lmo.iteration_counter
    # do stuff
    for _ in 1:5
        inner_iteration += 1
        # do stuff
    end
end

struct Agnostic <: FrankWolfe.LineSearchMethod end
FrankWolfe.perform_line_search(
    ls::Agnostic,
    t,
    f,
    g!,
    gradient,
    x,
    d,
    gamma_max,
    workspace,
    memory_mode,
) = 1 / sqrt(t+1)

#### EXAMPLE

function f(x)
    return -x[1] +2* (x[1]^2+x[2]^2-1)+1.75* abs(x[1]^2+x[2]^2-1)
end

function grad!(storage, x)
    function eval_a(x)
        return [-1+4*x[1], 4*x[2]]
    end
    function eval_cz(x)
        x[1]^2 + x[2]^2 -1
    end
    a = eval_a(x)
    b = [1.75]
    resz = eval_cz(x)
    #@show resz
    sigma_z = [sign.(resz)]
    c = vcat(a, b .* sigma_z)
    @show c
    @. storage = c
end

function func_init()
    return 2, 1
end

n, s = func_init()

# placeholder, has to be udapted to feasible region in chained_mifflin2
o = GLPK.Optimizer()
x = MOI.add_variables(o, n)
c1 = MOI.add_constraint(o, -1.0x[1] + x[2], MOI.LessThan(2.0))
c2 = MOI.add_constraint(o, x[1] + 2.0x[2], MOI.LessThan(4.0))
c3 = MOI.add_constraint(o, -2.0x[1] - x[2], MOI.LessThan(1.0))
c4 = MOI.add_constraint(o, x[1] - 2.0x[2], MOI.LessThan(2.0))
c5 = MOI.add_constraint(o, x[1] + 0.0x[2], MOI.LessThan(2.0))

lmo_moi = FrankWolfe.MathOptLMO(o)

x0 = [-1.0, -1-0]

# TODO: gradient does not match size of x, custom gradient passed
# TODO: dual gap ignored for now
# TODO: build LMO
# TODO: add asm to compute_extreme_point
x, v, primal, dual_gap, traj_data = as_frank_wolfe(
    f,
    grad!,
    lmo_moi,
    x0;
    gradient = ones(n+s),
    line_search = FrankWolfe.Agnostic(),
    verbose=true
)

@show x, v, primal