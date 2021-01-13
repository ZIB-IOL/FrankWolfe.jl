module FrankWolfe

using LinearAlgebra

include("defs.jl")
include("simplex_matrix.jl")
include("oracles.jl")
include("simplex_oracle.jl")
include("utils.jl")

# very simple FW Variant
# TODO:
# - need to remove unnecessary outputs (uses currently tabulate via pycall)
# - should support "silent mode" with no printing
# - should support "performance mode" with minimal computation and printing of intermediate information


function fw(f, grad, lmo, x0; stepSize::LSMethod = agnostic, 
        epsilon=1e-7, maxIt=10000, printIt=1000, trajectory=false, verbose=false) where T
    t = 0
    dualGap = Inf
    primal = Inf
    v = []
    x = x0
    trajData = []
    dx = similar(x0) # Array{eltype(x0)}(undef, length(x0))
    timeEl = time_ns()
    # for table printing // to be replaced
    #
    # headers = ["Type", "Iteration", "Primal", "Dual", "Dual Gap","Time"]
    # width = [6, 13, 14, 14, 14, 14]
    # if verbose
    #     println(tp.header(headers, width=width))
    # end
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
           nothing, gamma = segmentSearch(f,grad,x,v,lsTol=min(max(eps(),dualGap ^ 2,epsilon ^ 2),epsilon ^ (1/2))) 
        elseif stepSize === backtracking
           nothing, gamma = backtrackingLS(f,grad,x,v,lsTol=min(max(eps(),dualGap ^ 2,epsilon ^ 2),epsilon ^ (1/2))) 
        end
        
        # that's the expensive version in terms of memory but fast
        x = (1-gamma) * x + gamma * v
        
        if mod(t,printIt) == 0 && verbose
            tt = "FW"
            if t === 0
                tt = "I"
            end
            rep = [tt, string(t), primal, primal-dualGap, dualGap, (time_ns() - timeEl)/1.0e9]
            # timeEl = time_ns()
            # println(tp.row(rep,width=width,format_spec="5e"))
            flush(stdout)
        end
        t = t + 1
    end
    if verbose
        # println(tp.bottom(length(width),width=width))
        flush(stdout)
    end
    return x, v, primal, dualGap, trajData
end





end
