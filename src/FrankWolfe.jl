module FrankWolfe

using LinearAlgebra
using Printf
using Base.Ryu
using ProgressMeter
# using BenchmarkTools
using TimerOutputs

include("defs.jl")
include("simplex_matrix.jl")
include("oracles.jl")
include("simplex_oracle.jl")
include("utils.jl")


# simple beyond of core costs of algorithm

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
    print_timer(to::TimerOutput)
end

# very simple FW Variant
# TODO:
# - should support "performance mode" with minimal computation and printing of intermediate information


function fw(f, grad, lmo, x0; stepSize::LSMethod = agnostic, 
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

    if verbose
        if emph === blas
            println("\n EMPHASIS: blas")
        elseif emph === memory
            println("\n EMPHASIS: memory")
        end
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
        
        if stepSize === agnostic
            gamma = 2/(2+t)
        elseif stepSize === goldenratio
           nothing, gamma = segmentSearch(f,grad,x,v,lsTol=lsTol)
        elseif stepSize === backtracking
           nothing, gamma = backtrackingLS(f,grad,x,v,lsTol=lsTol) 
        end
        
        if emph === blas
            x = (1-gamma) * x + gamma * v
        elseif emph === memory
            @. x = (1-gamma) * x + gamma * v 
        end

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
        rep = [tt, string(t), primal, primal-dualGap, dualGap, (time_ns() - timeEl)/1.0e9]
        footerPrint()
        flush(stdout)
    end
    return x, v, primal, dualGap, trajData
end



end
