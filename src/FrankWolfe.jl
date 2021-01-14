module FrankWolfe

using LinearAlgebra
using Printf
using Base.Ryu
using ProgressMeter
using TimerOutputs

include("defs.jl")
include("simplex_matrix.jl")

include("oracles.jl")
include("simplex_oracle.jl")
include("lp_norm_oracles.jl")

include("utils.jl")

##############################################################
# simple benchmark of elementary costs of oracles and 
# critical components
##############################################################

function benchmarkOracles(f,grad,lmo,n;k=100,T=Float64)
    sv = n*sizeof(T)/1024/1024
    println("\nSize of single vector ($T): $sv MB\n")
    to = TimerOutput()
    @showprogress 1 "Testing f..." for i in 1:k    
        x = rand(n)
        @timeit to "f" temp = f(x)
    end
    @showprogress 1 "Testing grad..." for i in 1:k    
        x = rand(n)
        @timeit to "grad" temp = grad(x)
    end
    @showprogress 1 "Testing lmo..." for i in 1:k    
        x = rand(n)
        @timeit to "lmo" temp = compute_extreme_point(lmo, x)
    end
    @showprogress 1 "Testing dual gap..." for i in 1:k    
        @timeit to "dual gap" begin
            x = rand(n)
            gradient = grad(x)
            v = compute_extreme_point(lmo, gradient)
            dualGap = dot(x, gradient) - dot(v, gradient)
        end
    end
    @showprogress 1 "Testing update... (emph: blas)" for i in 1:k    
        x = rand(n)
        gradient = grad(x)
        v = compute_extreme_point(lmo, gradient)
        gamma = 1/2
        @timeit to "update (blas)" x = (1-gamma) * x + gamma * v
    end
    @showprogress 1 "Testing update... (emph: memory)" for i in 1:k    
        x = rand(n)
        gradient = grad(x)
        v = compute_extreme_point(lmo, gradient)
        gamma = 1/2
        # TODO: to be updated to broadcast version once data structure MaybeHotVector allows for it
        @timeit to "update (memory)" x = (1-gamma) * x + gamma * v
    end
    @showprogress 1 "Testing caching 100 points... " for i in 1:k    
        @timeit to "caching 100 points" begin
            cache = []
            for j in 1:100
                x = rand(n)
                push!(cache,x)
            end
            x = rand(n)
            gradient = grad(x)
            v = compute_extreme_point(lmo, gradient)
            gamma = 1/2    
            test = (x -> dot(x, gradient)).(cache) 
            v = cache[argmin(test)]
            val = v in cache
        end
    end

    print_timer(to::TimerOutput)
end

##############################################################
# Vanilla FW
##############################################################

function fw(f, grad, lmo, x0; stepSize::LSMethod = agnostic, L = Inf,
        stepLim=20,
        epsilon=1e-7, maxIt=10000, printIt=1000, trajectory=false, verbose=false,lsTol=1e-7,emph::Emph = blas) where T
    
    function headerPrint(data)
        @printf("\n───────────────────────────────────────────────────────────────────────────────────\n")
        @printf("%6s %13s %14s %14s %14s %14s\n", data[1], data[2], data[3], data[4], data[5], data[6])
        @printf("───────────────────────────────────────────────────────────────────────────────────\n")
    end

    function footerPrint()
        @printf("───────────────────────────────────────────────────────────────────────────────────\n\n")
    end
    
    function itPrint(data)
        @printf("%6s %13s %14e %14e %14e %14e\n", data[1], data[2], data[3], data[4], data[5], data[6])
    end

    t = 0
    dualGap = Inf
    primal = Inf
    v = []
    x = x0
    trajData = []
    dx = similar(x0) # Array{eltype(x0)}(undef, length(x0))
    timeEl = time_ns()
    
    if stepSize === shortstep && L == Inf
        println("WARNING: Lipschitz constant not set. Prepare to blow up spectacularly.")
    end

    if verbose
        println("\nVanilla Frank-Wolfe Algorithm.")
        println("EMPHASIS: $emph STEPSIZE: $stepSize EPSILON: $epsilon MAXIT: $maxIt")
        headers = ["Type", "Iteration", "Primal", "Dual", "Dual Gap","Time"]
        headerPrint(headers)
    end

    while t <= maxIt && dualGap >= max(epsilon,eps())
        primal = f(x)
        gradient = grad(x)
        v = compute_extreme_point(lmo, gradient)
        
        dualGap = dot(x, gradient) - dot(v, gradient)

        if trajectory === true
            append!(trajData, [t, primal, primal-dualGap, dualGap])
        end
 
        if trajectory === true
            append!(trajData, [t, primal, primal-dualGap, dualGap, (time_ns() - timeEl)/1.0e9])
        end
    
        if stepSize === agnostic
            gamma = 2/(2+t)
        elseif stepSize === goldenratio
           nothing, gamma = segmentSearch(f,grad,x,v,lsTol=lsTol)
        elseif stepSize === backtracking
           nothing, gamma = backtrackingLS(f,grad,x,v,lsTol=lsTol,stepLim=stepLim) 
        elseif stepSize === nonconvex
            gamma = 1 / sqrt(t+1)
        elseif stepSize === shortstep
            gamma = dualGap / (L * norm(x-v)^2 )
        end

        @emphasis(emph, x = (1-gamma) * x + gamma * v)

        if mod(t,printIt) == 0 && verbose
            tt = "FW"
            if t === 0
                tt = "I"
            end
            rep = [tt, string(t), primal, primal-dualGap, dualGap, (time_ns() - timeEl)/1.0e9]
            itPrint(rep)
            flush(stdout)
        end
        t = t + 1
    end
    if verbose
        tt = "Last"
        rep = [tt, "" , primal, primal-dualGap, dualGap, (time_ns() - timeEl)/1.0e9]
        itPrint(rep)
        footerPrint()
        flush(stdout)
    end
    return x, v, primal, dualGap, trajData
end

##############################################################
# Lazified Vanilla FW
##############################################################

function lcg(f, grad, lmo, x0; stepSize::LSMethod = agnostic, L = Inf,
    phiFactor=2,
    epsilon=1e-7, maxIt=10000, printIt=1000, trajectory=false, verbose=false,lsTol=1e-7,emph::Emph = blas) where T

function headerPrint(data)
    @printf("\n─────────────────────────────────────────────────────────────────────────────────────────────────\n")
    @printf("%6s %13s %14s %14s %14s %14s %14s\n", data[1], data[2], data[3], data[4], data[5], data[6], data[7])
    @printf("─────────────────────────────────────────────────────────────────────────────────────────────────\n")
end

function footerPrint()
    @printf("─────────────────────────────────────────────────────────────────────────────────────────────────\n\n")
end

function itPrint(data)
    @printf("%6s %13s %14e %14e %14e %14e %14s\n", data[1], data[2], data[3], data[4], data[5], data[6], data[7])
end

t = 0
dualGap = Inf
primal = Inf
v = []
x = x0
phi = Inf
cache = []
trajData = []
dx = similar(x0) # Array{eltype(x0)}(undef, length(x0))
timeEl = time_ns()

if stepSize === shortstep && L == Inf
    println("WARNING: Lipschitz constant not set. Prepare to blow up spectacularly.")
end

if stepSize === agnostic || stepSize === nonconvex
    println("WARNING: Lazification is not known to converge with open-loop step size strategies.")
end

if verbose
    println("\nLazified Conditional Gradients (Frank-Wolfe + Lazification).")
    println("EMPHASIS: $emph STEPSIZE: $stepSize EPSILON: $epsilon MAXIT: $maxIt PHIFACTOR: $phiFactor")
    headers = ["Type", "Iteration", "Primal", "Dual", "Dual Gap","Time", "Cache Size"]
    headerPrint(headers)
end

while t <= maxIt && dualGap >= max(epsilon,eps())

    primal = f(x)
    gradient = grad(x)

    if !isempty(cache)
        tt = "L"
        # let us be optimistic first:
        if dot(x,gradient) - dot(v,gradient) < phi  # only look up new point if old one is not good enough
            test = (x -> dot(x, gradient)).(cache) # TODO: also needs to have a broadcast version as can be expensive
            v = cache[argmin(test)]
        end
        if dot(x,gradient) - dot(v,gradient) < phi # still not good enough, then solve the LP
            tt = "FW"
            v = compute_extreme_point(lmo, gradient)
            dualGap = dot(x,gradient) - dot(v,gradient)
            if dualGap < phi # still not good enough, then we have a proof of halving
                phi = dualGap / 2
            end
            if !(v in cache) 
                push!(cache,v)
            end
        end
    else    
        v = compute_extreme_point(lmo, gradient)
        dualGap = dot(x,gradient) - dot(v,gradient)
        phi = dualGap / 2
        push!(cache,v)
    end
    
    if trajectory === true
        append!(trajData, [t, primal, primal-dualGap, dualGap, (time_ns() - timeEl)/1.0e9, length(cache)])
    end
    
    if stepSize === agnostic
        gamma = 2/(2+t)
    elseif stepSize === goldenratio
       nothing, gamma = segmentSearch(f,grad,x,v,lsTol=lsTol)
    elseif stepSize === backtracking
       nothing, gamma = backtrackingLS(f,grad,x,v,lsTol=lsTol) 
    elseif stepSize === nonconvex
        gamma = 1 / sqrt(t+1)
    elseif stepSize === shortstep
        gamma = dualGap / (L * norm(x-v)^2 )
    end

    @emphasis(emph, x = (1.0 -gamma) * x + gamma * v)

    if mod(t,printIt) == 0 && verbose
        tt = "FW"
        if t === 0
            tt = "I"
        end
        rep = [tt, string(t), primal, primal-dualGap, dualGap, (time_ns() - timeEl)/1.0e9, length(cache)]
        itPrint(rep)
        flush(stdout)
    end
    t = t + 1
end
if verbose
    tt = "Last"
    rep = [tt, "", primal, primal-dualGap, dualGap, (time_ns() - timeEl)/1.0e9, length(cache)]
    itPrint(rep)
    footerPrint()
    flush(stdout)
end
return x, v, primal, dualGap, trajData
end


end
