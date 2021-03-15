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

import Arpack
using DoubleFloats

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
    emphasis::Emphasis=memory,
    nep=false,
    gradient=nothing
)
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
            data[7]        )
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
            "%6s %13s %14e %14e %14e %14e %14e\n",
            st[Symbol(data[1])],
            data[2],
            Float64(data[3]),
            Float64(data[4]),
            Float64(data[5]),
            data[6],
            data[7]
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

    if !isnothing(momentum) && (line_search === shortstep || line_search === adaptive || line_search === rationalshortstep)
        println("WARNING: Momentum-averaged gradients should usually be used with agnostic stepsize rules.")
    end

    if verbose
        println("\nVanilla Frank-Wolfe Algorithm.")
        numType = eltype(x0)
        println(
            "EMPHASIS: $emphasis STEPSIZE: $line_search EPSILON: $epsilon MAXITERATION: $max_iteration TYPE: $numType",
        )
        grad_type = typeof(gradient)
        println("MOMENTUM: $momentum GRADIENTTYPE: $grad_type")
        if emphasis === memory
            println("WARNING: In memory emphasis mode iterates are written back into x0!")
        end
        headers = ["Type", "Iteration", "Primal", "Dual", "Dual Gap", "Time", "It/sec"]
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

    # container for direction
    d = similar(x)

    gtemp = if momentum === nothing
        nothing
    else
        similar(x)
    end
    while t <= max_iteration && dual_gap >= max(epsilon, eps())
        if momentum === nothing || first_iter
            grad!(gradient, x)
            if momentum !== nothing
                gtemp .= gradient
            end
        else
            grad!(gtemp, x)
            @emphasis(emphasis, gradient = (momentum * gradient) + (1 - momentum) * gtemp)
        end
        first_iter = false
        
        # build-in NEP here
        if nep
            # argmin_v v^T(1-2y)
            # y = x_t - 1/L * (t+1)/2 * gradient
            # check whether emphasis works
            @emphasis(emphasis, gradient = 1 - 2 * (x - 1 / L * (t+1) / 2 * gradient))
        end

        v = compute_extreme_point(lmo, gradient)
        # go easy on the memory - only compute if really needed
        if (
            (mod(t, print_iter) == 0 && verbose) ||
            trajectory ||
            line_search == shortstep
        )
            primal = f(x)
            dual_gap = fast_dot(x, gradient) - fast_dot(v, gradient)
        end

        if trajectory
            push!(
                trajData,
                (t, primal, primal - dual_gap, dual_gap, (time_ns() - time_start) / 1.0e9),
            )
        end
        @emphasis(emphasis, d = x - v)

        if isnothing(momentum)
            gamma, L = line_search_wrapper(line_search,t,f,grad!,x, d,gradient,dual_gap,L,gamma0,linesearch_tol,step_lim, 1.0)
        else
            gamma, L = line_search_wrapper(line_search,t,f,grad!,x, d,gtemp,dual_gap,L,gamma0,linesearch_tol,step_lim, 1.0)
        end

        @emphasis(emphasis, x = x - gamma*d)

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
                t / ( (time_ns() - time_start) / 1.0e9 )
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
    dual_gap = fast_dot(x, gradient) - fast_dot(v, gradient)
    if verbose
        tt = last
        rep = (
            tt,
            string(t - 1),
            primal,
            primal - dual_gap,
            dual_gap,
            (time_ns() - time_start) / 1.0e9,
            t / ( (time_ns() - time_start) / 1.0e9 )
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
    lmo_base,
    x0;
    line_search::LineSearchMethod=agnostic,
    L=Inf,
    gamma0=0,
    phiFactor=2,
    cache_size=Inf,
    greedy_lazy=false,
    epsilon=1e-7,
    max_iteration=10000,
    print_iter=1000,
    trajectory=false,
    verbose=false,
    linesearch_tol=1e-7,
    step_lim=20,
    emphasis::Emphasis=memory,
    gradient=nothing,
    VType=typeof(x0),
)

    if isfinite(cache_size)
        lmo = MultiCacheLMO{cache_size, typeof(lmo_base), VType}(lmo_base)
    else
        lmo = VectorCacheLMO{typeof(lmo_base),VType}(lmo_base)
    end

    function print_header(data)
        @printf(
            "\n───────────────────────────────────────────────────────────────────────────────────────────────────────────────\n"
        )
        @printf(
            "%6s %13s %14s %14s %14s %14s %14s %14s\n",
            data[1],
            data[2],
            data[3],
            data[4],
            data[5],
            data[6],
            data[7],
            data[8]
        )
        @printf(
            "───────────────────────────────────────────────────────────────────────────────────────────────────────────────\n"
        )
    end

    function print_footer()
        @printf(
            "───────────────────────────────────────────────────────────────────────────────────────────────────────────────\n\n"
        )
    end

    function print_iter_func(data)
        @printf(
            "%6s %13s %14e %14e %14e %14e %14e %14s\n",
            st[Symbol(data[1])],
            data[2],
            data[3],
            data[4],
            data[5],
            data[6],
            data[7],
            data[8]
        )
    end

    t = 0
    dual_gap = Inf
    primal = Inf
    v = []
    x = x0
    phi = Inf
    trajData = []
    tt = regular
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
            "EMPHASIS: $emphasis STEPSIZE: $line_search EPSILON: $epsilon MAXITERATION: $max_iteration PHIFACTOR: $phiFactor TYPE: $numType",
        )
        println("cache_size $cache_size GREEDYCACHE: $greedy_lazy")
        if emphasis == memory
            println("WARNING: In memory emphasis mode iterates are written back into x0!")
        end
        headers = ["Type", "Iteration", "Primal", "Dual", "Dual Gap", "Time", "It/sec", "Cache Size"]
        print_header(headers)
    end

    if emphasis == memory && !isa(x, Union{Array, SparseArrays.AbstractSparseArray})
        x = convert(Array{float(eltype(x))}, x)
    end

    if gradient === nothing
        gradient = similar(x)
    end

    # container for direction
    d = similar(x)

    while t <= max_iteration && dual_gap >= max(epsilon, eps())

        grad!(gradient, x)

        threshold = fast_dot(x, gradient) - phi

        # go easy on the memory - only compute if really needed
        if (
            (mod(t, print_iter) == 0 && verbose) ||
            trajectory
        )
            primal = f(x)
        end

        v = compute_extreme_point(lmo, gradient, threshold=threshold, greedy=greedy_lazy)
        tt = lazy
        if fast_dot(v, gradient) > threshold
            tt = dualstep
            dual_gap = fast_dot(x, gradient) - fast_dot(v, gradient)
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

        @emphasis(emphasis, d = x - v)
        
        gamma, L = line_search_wrapper(line_search,t,f,grad!,x,d,gradient,dual_gap,L,gamma0,linesearch_tol,step_lim, 1.0)

        @emphasis(emphasis, x = x - gamma*d)

        if verbose && (mod(t, print_iter) == 0 || tt == dualstep)
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
                t / ( (time_ns() - time_start) / 1.0e9 ),
                length(lmo),
            )
            print_iter_func(rep)
            flush(stdout)
        end
        t += 1
    end

    # recompute everything once for final verfication / do not record to trajectory though for now! 
    # this is important as some variants do not recompute f(x) and the dual_gap regularly but only when reporting
    # hence the final computation.
    grad!(gradient, x)
    v = compute_extreme_point(lmo, gradient)
    primal = f(x)
    dual_gap = fast_dot(x, gradient) - fast_dot(v, gradient)

    if verbose
        tt = last
        rep = (
            tt,
            string(t - 1),
            primal,
            primal - dual_gap,
            dual_gap,
            (time_ns() - time_start) / 1.0e9,
            t / ( (time_ns() - time_start) / 1.0e9 ),
            length(lmo),
        )
        print_iter_func(rep)
        print_footer()
        flush(stdout)
    end
    return x, v, primal, dual_gap, trajData
end

end
