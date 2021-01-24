module FrankWolfe

using LinearAlgebra
using Printf
using ProgressMeter
using TimerOutputs
using SparseArrays: spzeros
import Random

# for plotting -> keep here or move somewhere else?
using Plots

include("defs.jl")
include("simplex_matrix.jl")

include("oracles.jl")
include("simplex_oracles.jl")
include("lp_norm_oracles.jl")
include("polytope_oracles.jl")

include("utils.jl")
include("function_gradient.jl")
include("active_set.jl")

include("blended_cg.jl")

# move advanced variants etc to there own files to prevent excessive clutter
include("afw.jl")

include("fw_algorithms.jl")

##############################################################
# simple benchmark of elementary costs of oracles and 
# critical components
##############################################################

# TODO: add actual use of T for the rand(n)

function benchmark_oracles(f, grad, lmo, n; k=100, nocache=true, T=Float64)
    sv = n * sizeof(T) / 1024 / 1024
    println("\nSize of single vector ($T): $sv MB\n")
    to = TimerOutput()
    @showprogress 1 "Testing f... " for i in 1:k
        x = rand(n)
        @timeit to "f" temp = f(x)
    end
    @showprogress 1 "Testing grad... " for i in 1:k
        x = rand(n)
        @timeit to "grad" temp = grad(x)
    end
    @showprogress 1 "Testing lmo... " for i in 1:k
        x = rand(n)
        @timeit to "lmo" temp = compute_extreme_point(lmo, x)
    end
    @showprogress 1 "Testing dual gap... " for i in 1:k
        x = rand(n)
        gradient = grad(x)
        v = compute_extreme_point(lmo, gradient)
        @timeit to "dual gap" begin
            dualGap = dot(x, gradient) - dot(v, gradient)
        end
    end
    @showprogress 1 "Testing update... (Emphasis: blas) " for i in 1:k
        x = rand(n)
        gradient = grad(x)
        v = compute_extreme_point(lmo, gradient)
        gamma = 1 / 2
        @timeit to "update (blas)" @emphasis(blas, x = (1 - gamma) * x + gamma * v)
    end
    @showprogress 1 "Testing update... (Emphasis: memory) " for i in 1:k
        x = rand(n)
        gradient = grad(x)
        v = compute_extreme_point(lmo, gradient)
        gamma = 1 / 2
        # TODO: to be updated to broadcast version once data structure MaybeHotVector allows for it
        @timeit to "update (memory)" @emphasis(memory, x = (1 - gamma) * x + gamma * v)
    end
    if !nocache
        @showprogress 1 "Testing caching 100 points... " for i in 1:k
            @timeit to "caching 100 points" begin
                cache = Float64[]
                for j in 1:100
                    x = rand(n)
                    push!(cache, x)
                end
                x = rand(n)
                gradient = grad(x)
                v = compute_extreme_point(lmo, gradient)
                gamma = 1 / 2
                test = (x -> dot(x, gradient)).(cache)
                v = cache[argmin(test)]
                val = v in cache
            end
        end
    end
    print_timer(to)
    return nothing
end

##############################################################
# Vanilla FW
##############################################################

function fw(
    f,
    grad,
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
    Emphasis::Emphasis=blas,
)
    function headerPrint(data)
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

    function footerPrint()
        @printf(
            "───────────────────────────────────────────────────────────────────────────────────\n\n"
        )
    end

    function itPrint(data)
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
    dualGap = Inf
    primal = Inf
    v = []
    x = x0
    tt:StepType = regular
    trajData = []
    dx = similar(x0) # Array{eltype(x0)}(undef, length(x0))
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
            "EMPHASIS: $Emphasis STEPSIZE: $line_search EPSILON: $epsilon max_iteration: $max_iteration TYPE: $numType",
        )
        println("MOMENTUM: $momentum")
        if Emphasis === memory
            println("WARNING: In memory emphasis mode iterates are written back into x0!")
        end
        headers = ["Type", "Iteration", "Primal", "Dual", "Dual Gap", "Time"]
        headerPrint(headers)
    end

    if Emphasis === memory && !isa(x, Array)
        x = convert(Vector{promote_type(eltype(x), Float64)}, x)
    end
    first_iter = true
    gradient = 0
    while t <= max_iteration && dualGap >= max(epsilon, eps())

        if momentum === nothing || first_iter
            gradient = grad(x)
        else
            @emphasis(Emphasis, gradient = (momentum * gradient) .+ (1 - momentum) .* grad(x))
        end
        first_iter = false

        v = compute_extreme_point(lmo, gradient)

        # go easy on the memory - only compute if really needed
        if (
            (mod(t, print_iter) == 0 && verbose) ||
            trajectory ||
            !(line_search == agnostic || line_search == nonconvex || line_search == fixed)
        )
            primal = f(x)
            dualGap = dot(x, gradient) - dot(v, gradient)
        end

        if trajectory === true
            push!(trajData, [t, primal, primal - dualGap, dualGap, (time_ns() - time_start) / 1.0e9])
        end

        if line_search === agnostic
            gamma = 2 // (2 + t)
        elseif line_search === goldenratio
            _, gamma = segmentSearch(f, grad, x, v, linesearch_tol=linesearch_tol)
        elseif line_search === backtracking
            _, gamma = backtrackingLS(f, grad, x, v, linesearch_tol=linesearch_tol, step_lim=step_lim)
        elseif line_search === nonconvex
            gamma = 1 / sqrt(t + 1)
        elseif line_search === shortstep
            gamma = dualGap / (L * norm(x - v)^2)
        elseif line_search === rationalshortstep
            ratDualGap = sum((x - v) .* gradient)
            gamma = ratDualGap // (L * sum((x - v) .^ 2))
        elseif line_search === fixed
            gamma = gamma
        elseif line_search === adaptive
            L, gamma = adaptive_step_size(f, gradient, x, x - v, L)
        end

        @emphasis(Emphasis, x = (1 - gamma) * x + gamma * v)

        if mod(t, print_iter) == 0 && verbose
            tt = regular
            if t === 0
                tt = initial
            end
            rep = [tt, string(t), primal, primal - dualGap, dualGap, (time_ns() - time_start) / 1.0e9]
            itPrint(rep)
            flush(stdout)
        end
        t = t + 1
    end
    # recompute everything once for final verfication / do not record to trajectory though for now! 
    # this is important as some variants do not recompute f(x) and the dualGap regularly but only when reporting
    # hence the final computation.
    gradient = grad(x)
    v = compute_extreme_point(lmo, gradient)
    primal = f(x)
    dualGap = dot(x, gradient) - dot(v, gradient)
    if verbose
        tt = last
        rep = [tt, string(t - 1), primal, primal - dualGap, dualGap, (time_ns() - time_start) / 1.0e9]
        itPrint(rep)
        footerPrint()
        flush(stdout)
    end
    return x, v, primal, dualGap, trajData
end

##############################################################
# Lazified Vanilla FW
##############################################################

function lcg(
    f,
    grad,
    lmoBase,
    x0;
    line_search::LineSearchMethod=agnostic,
    L=Inf,
    phiFactor=2,
    cacheSize=Inf,
    greedyLazy=false,
    epsilon=1e-7,
    max_iteration=10000,
    print_iter=1000,
    trajectory=false,
    verbose=false,
    linesearch_tol=1e-7,
    Emphasis::Emphasis=blas,
)

    if isfinite(cacheSize)
        lmo = MultiCacheLMO{cacheSize}(lmoBase)
    else
        lmo = VectorCacheLMO(lmoBase)
    end

    function headerPrint(data)
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

    function footerPrint()
        @printf(
            "─────────────────────────────────────────────────────────────────────────────────────────────────\n\n"
        )
    end

    function itPrint(data)
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
    dualGap = Inf
    primal = Inf
    v = []
    x = x0
    phi = Inf
    trajData = []
    tt::StepType = regular
    dx = similar(x0) # Array{eltype(x0)}(undef, length(x0))
    time_start = time_ns()

    if line_search === shortstep && L == Inf
        println("FATAL: Lipschitz constant not set. Prepare to blow up spectacularly.")
    end

    if line_search === agnostic || line_search === nonconvex
        println("FATAL: Lazification is not known to converge with open-loop step size strategies.")
    end

    if verbose
        println("\nLazified Conditional Gradients (Frank-Wolfe + Lazification).")
        numType = eltype(x0)
        println(
            "EMPHASIS: $Emphasis STEPSIZE: $line_search EPSILON: $epsilon max_iteration: $max_iteration PHIFACTOR: $phiFactor TYPE: $numType",
        )
        println("CACHESIZE $cacheSize GREEDYCACHE: $greedyLazy")
        if Emphasis === memory
            println("WARNING: In memory emphasis mode iterates are written back into x0!")
        end
        headers = ["Type", "Iteration", "Primal", "Dual", "Dual Gap", "Time", "Cache Size"]
        headerPrint(headers)
    end

    if Emphasis === memory && !isa(x, Array)
        x = convert(Vector{promote_type(eltype(x), Float64)}, x)
    end

    while t <= max_iteration && dualGap >= max(epsilon, eps())

        primal = f(x)
        gradient = grad(x)

        threshold = dot(x, gradient) - phi

        v = compute_extreme_point(lmo, gradient, threshold=threshold, greedy=greedyLazy)
        tt = lazy
        if dot(v, gradient) > threshold
            tt = dualstep
            dualGap = dot(x, gradient) - dot(v, gradient)
            phi = dualGap / 2
        end

        if trajectory === true
            push!(
                trajData,
                [t, primal, primal - dualGap, dualGap, (time_ns() - time_start) / 1.0e9, length(lmo)],
            )
        end

        if line_search === agnostic
            gamma = 2 / (2 + t)
        elseif line_search === goldenratio
            _, gamma = segmentSearch(f, grad, x, v, linesearch_tol=linesearch_tol)
        elseif line_search === backtracking
            _, gamma = backtrackingLS(f, grad, x, v, linesearch_tol=linesearch_tol)
        elseif line_search === nonconvex
            gamma = 1 / sqrt(t + 1)
        elseif line_search === shortstep
            gamma = dualGap / (L * dot(x - v, x - v))
        end

        @emphasis(Emphasis, x = (1 - gamma) * x + gamma * v)

        if mod(t, print_iter) == 0 || tt == dualstep && verbose
            if t === 0
                tt = initial
            end
            rep = [
                tt,
                string(t),
                primal,
                primal - dualGap,
                dualGap,
                (time_ns() - time_start) / 1.0e9,
                length(lmo),
            ]
            itPrint(rep)
            flush(stdout)
        end
        t = t + 1
    end
    if verbose
        tt = last
        rep = [
            tt,
            string(t - 1),
            primal,
            primal - dualGap,
            dualGap,
            (time_ns() - time_start) / 1.0e9,
            length(lmo),
        ]
        itPrint(rep)
        footerPrint()
        flush(stdout)
    end
    return x, v, primal, dualGap, trajData
end


end
