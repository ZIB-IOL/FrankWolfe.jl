module FrankWolfe

using LinearAlgebra
using Printf
using ProgressMeter
using TimerOutputs
using SparseArrays: spzeros, SparseVector
import SparseArrays
import Random

import MathOptInterface
const MOI = MathOptInterface

# for plotting -> keep here or move somewhere else?
using Plots

# for Birkhoff polytope LMO
import Hungarian

# for nuclear norm
import IterativeSolvers

include("defs.jl")
include("simplex_matrix.jl")

include("utils.jl")
include("oracles.jl")
include("simplex_oracles.jl")
include("norm_oracles.jl")
include("polytope_oracles.jl")
include("moi_oracle.jl")
include("function_gradient.jl")
include("active_set.jl")

# move advanced variants etc to their own files to prevent excessive clutter

include("blended_cg.jl")
include("afw.jl")
include("fw_algorithms.jl")

##############################################################
# Vanilla FW
##############################################################

function fw(
    f,
    grad!,
    lmo,
    x0;
    line_search::LineSearchMethod=agnostic,
    L=Inf,
    gamma0=0,
    step_lim=20,
    momentum=nothing,
    epsilon=1e-7,
    max_iteration=10000,
    print_iter=1000,
    trajectory=false,
    verbose=false,
    linesearch_tol=1e-7,
    emphasis::Emphasis=blas,
    nep=false,
    gradient=nothing
)
    function print_header(data)
        @printf(
            "\n───────────────────────────────────────────────────────────────────────────────────\n"
        )
        @printf(
            "%6s %13s %14s %14s %14s %14s\n",
            data[1],
            data[2],
            data[3],
            data[4],
            data[5],
            data[6]
        )
        @printf(
            "───────────────────────────────────────────────────────────────────────────────────\n"
        )
    end

    function print_footer()
        @printf(
            "───────────────────────────────────────────────────────────────────────────────────\n\n"
        )
    end

    function print_iter_func(data)
        @printf(
            "%6s %13s %14e %14e %14e %14e\n",
            st[Symbol(data[1])],
            data[2],
            Float64(data[3]),
            Float64(data[4]),
            Float64(data[5]),
            data[6]
        )
    end

    t = 0
    dual_gap = Inf
    primal = Inf
    v = []
    x = x0
    tt = regular
    trajData = []
    time_start = time_ns()

    if (line_search === shortstep || line_search === adaptive) && L == Inf
        println("FATAL: Lipschitz constant not set. Prepare to blow up spectacularly.")
    end

    if line_search === fixed && gamma0 == 0
        println("FATAL: gamma0 not set. We are not going to move a single bit.")
    end

    if verbose
        println("\nVanilla Frank-Wolfe Algorithm.")
        numType = eltype(x0)
        println(
            "EMPHASIS: $emphasis STEPSIZE: $line_search EPSILON: $epsilon max_iteration: $max_iteration TYPE: $numType",
        )
        println("MOMENTUM: $momentum")
        if emphasis === memory
            println("WARNING: In memory emphasis mode iterates are written back into x0!")
        end
        headers = ["Type", "Iteration", "Primal", "Dual", "Dual Gap", "Time"]
        print_header(headers)
    end

    if emphasis === memory && !isa(x, Array)
        x = convert(Array{promote_type(eltype(x), Float64)}, x)
    end
    first_iter = true
    # instanciating container for gradient
    if gradient === nothing
        gradient = similar(x)
    end
    gtemp = if momentum === nothing
        nothing
    else
        similar(x)
    end
    while t <= max_iteration && dual_gap >= max(epsilon, eps())
        if momentum === nothing || first_iter
            grad!(gradient, x)
        else
            grad!(gtemp, x)
            @emphasis(emphasis, gradient = (momentum * gradient) + (1 - momentum) * gtemp)
        end
        first_iter = false
        
        # build-in NEP here
        if nep === true
            # argmin_v v^T(1-2y)
            # y = x_t - 1/L * (t+1)/2 * gradient
            @. gradient = 1 - 2 * (x - 1 / L * (t+1) / 2 * gradient)
        end

        v = compute_extreme_point(lmo, gradient)

        # go easy on the memory - only compute if really needed
        if (
            (mod(t, print_iter) == 0 && verbose) ||
            trajectory ||
            !(line_search == agnostic || line_search == nonconvex || line_search == fixed)
        )
            primal = f(x)
            dual_gap = dot(x, gradient) - dot(v, gradient)
        end

        if trajectory
            push!(
                trajData,
                (t, primal, primal - dual_gap, dual_gap, (time_ns() - time_start) / 1.0e9),
            )
        end

        if line_search === agnostic
            gamma = 2 // (2 + t)
        elseif line_search === goldenratio
            _, gamma = segment_search(f, grad!, x, v, linesearch_tol=linesearch_tol, inplace_gradient=true)
        elseif line_search === backtracking
            _, gamma =
                backtrackingLS(f, gradient, x, v, linesearch_tol=linesearch_tol, step_lim=step_lim)
        elseif line_search === nonconvex
            gamma = 1 / sqrt(t + 1)
        elseif line_search === shortstep
            gamma = dual_gap / (L * norm(x - v)^2)
        elseif line_search === rationalshortstep
            rat_dual_gap = sum((x - v) .* gradient)
            gamma = rat_dual_gap // (L * sum((x - v) .^ 2))
        elseif line_search === fixed
            gamma = gamma
        elseif line_search === adaptive
            L, gamma = adaptive_step_size(f, gradient, x, x - v, L)
        end
        @debug "gamma: $gamma"

        @emphasis(emphasis, x = (1 - gamma) * x + gamma * v)

        if mod(t, print_iter) == 0 && verbose
            tt = regular
            if t == 0
                tt = initial
            end
            rep = (
                tt,
                string(t),
                primal,
                primal - dual_gap,
                dual_gap,
                (time_ns() - time_start) / 1.0e9,
            )
            print_iter_func(rep)
            flush(stdout)
        end
        t = t + 1
    end
    # recompute everything once for final verfication / do not record to trajectory though for now! 
    # this is important as some variants do not recompute f(x) and the dual_gap regularly but only when reporting
    # hence the final computation.
    grad!(gradient, x)
    v = compute_extreme_point(lmo, gradient)
    primal = f(x)
    dual_gap = dot(x, gradient) - dot(v, gradient)
    if verbose
        tt = last
        rep = (
            tt,
            string(t - 1),
            primal,
            primal - dual_gap,
            dual_gap,
            (time_ns() - time_start) / 1.0e9,
        )
        print_iter_func(rep)
        print_footer()
        flush(stdout)
    end
    return x, v, primal, dual_gap, trajData
end

##############################################################
# Lazified Vanilla FW
##############################################################

function lcg(
    f,
    grad!,
    lmoBase,
    x0;
    line_search::LineSearchMethod=agnostic,
    L=Inf,
    phiFactor=2,
    cache_size=Inf,
    greedy_lazy=false,
    epsilon=1e-7,
    max_iteration=10000,
    print_iter=1000,
    trajectory=false,
    verbose=false,
    linesearch_tol=1e-7,
    emphasis::Emphasis=blas,
    gradient=nothing,
)

    if isfinite(cache_size)
        lmo = MultiCacheLMO{cache_size}(lmoBase)
    else
        lmo = VectorCacheLMO(lmoBase)
    end

    function print_header(data)
        @printf(
            "\n─────────────────────────────────────────────────────────────────────────────────────────────────\n"
        )
        @printf(
            "%6s %13s %14s %14s %14s %14s %14s\n",
            data[1],
            data[2],
            data[3],
            data[4],
            data[5],
            data[6],
            data[7]
        )
        @printf(
            "─────────────────────────────────────────────────────────────────────────────────────────────────\n"
        )
    end

    function print_footer()
        @printf(
            "─────────────────────────────────────────────────────────────────────────────────────────────────\n\n"
        )
    end

    function print_iter_func(data)
        @printf(
            "%6s %13s %14e %14e %14e %14e %14s\n",
            st[Symbol(data[1])],
            data[2],
            data[3],
            data[4],
            data[5],
            data[6],
            data[7]
        )
    end

    t = 0
    dual_gap = Inf
    primal = Inf
    v = []
    x = x0
    phi = Inf
    trajData = []
    tt::StepType = regular
    time_start = time_ns()

    if line_search == shortstep && L == Inf
        println("FATAL: Lipschitz constant not set. Prepare to blow up spectacularly.")
    end

    if line_search == agnostic || line_search == nonconvex
        println("FATAL: Lazification is not known to converge with open-loop step size strategies.")
    end

    if verbose
        println("\nLazified Conditional Gradients (Frank-Wolfe + Lazification).")
        numType = eltype(x0)
        println(
            "EMPHASIS: $emphasis STEPSIZE: $line_search EPSILON: $epsilon max_iteration: $max_iteration PHIFACTOR: $phiFactor TYPE: $numType",
        )
        println("cache_size $cache_size GREEDYCACHE: $greedy_lazy")
        if emphasis == memory
            println("WARNING: In memory emphasis mode iterates are written back into x0!")
        end
        headers = ["Type", "Iteration", "Primal", "Dual", "Dual Gap", "Time", "Cache Size"]
        print_header(headers)
    end

    if emphasis == memory && !isa(x, Union{Array, SparseArrays.AbstractSparseArray})
        x = convert(Array{promote_type(eltype(x), Float64)}, x)
    end

    if gradient === nothing
        gradient = similar(x)
    end

    while t <= max_iteration && dual_gap >= max(epsilon, eps())

        primal = f(x)
        grad!(gradient, x)

        threshold = dot(x, gradient) - phi

        v = compute_extreme_point(lmo, gradient, threshold=threshold, greedy=greedy_lazy)
        tt = lazy
        if dot(v, gradient) > threshold
            tt = dualstep
            dual_gap = dot(x, gradient) - dot(v, gradient)
            phi = dual_gap / 2
        end

        if trajectory
            push!(
                trajData,
                (
                    t,
                    primal,
                    primal - dual_gap,
                    dual_gap,
                    (time_ns() - time_start) / 1.0e9,
                    length(lmo),
                ),
            )
        end

        if line_search == agnostic
            gamma = 2 / (2 + t)
        elseif line_search == goldenratio
            _, gamma = segment_search(f, grad!, x, v, linesearch_tol=linesearch_tol)
        elseif line_search == backtracking
            _, gamma = backtrackingLS(f, gradient, x, v, linesearch_tol=linesearch_tol)
        elseif line_search == nonconvex
            gamma = 1 / sqrt(t + 1)
        elseif line_search == shortstep
            gamma = dot(gradient, x - v) / (L * dot(x - v, x - v))
        end

        @emphasis(emphasis, x = (1 - gamma) * x + gamma * v)

        if mod(t, print_iter) == 0 || tt == dualstep && verbose
            if t === 0
                tt = initial
            end
            rep = (
                tt,
                string(t),
                primal,
                primal - dual_gap,
                dual_gap,
                (time_ns() - time_start) / 1.0e9,
                length(lmo),
            )
            print_iter_func(rep)
            flush(stdout)
        end
        t = t + 1
    end
    if verbose
        tt = last
        rep = (
            tt,
            string(t - 1),
            primal,
            primal - dual_gap,
            dual_gap,
            (time_ns() - time_start) / 1.0e9,
            length(lmo),
        )
        print_iter_func(rep)
        print_footer()
        flush(stdout)
    end
    return x, v, primal, dual_gap, trajData
end

end
