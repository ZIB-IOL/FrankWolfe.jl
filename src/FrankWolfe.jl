module FrankWolfe

# Write your package code here.
include("simplex_matrix.jl")
include("utils.jl")

# very simple FW Variant
# TODO:
# - need to remove unnecessary outputs (uses currently tabulate via pycall)
# - should support "silent mode" with no printing
# - should support "performance mode" with minimal computation and printing of intermediate information


function fw(f, grad, lmo, x0::Vector{Float64}; stepSize = "agnostic", 
        epsilon=1e-7, maxIt=10000, printIt=1000, trajectory=false, silent=false)
    t = 0
    dualGap = Inf
    primal = Inf
    v = []
    x = x0
    trajData = []
    dx = Array{Float64}(undef, length(x0))
    timeEl = time_ns()
    # for table printing
    headers = ["Type", "Iteration", "Primal", "Dual", "Dual Gap","Time"]
    width = [6, 13, 14, 14, 14, 14]
    
    if !silent
        println(tp.header(headers, width=width))
    end
    while t <= maxIt && dualGap >= max(epsilon,eps())
        primal = f(x)
        gradient = grad(x)
        v = lmo(gradient)
        dualGap = dot(gradient,x) - dot(gradient,v)
        
        if trajectory === true
            append!(trajData, [t, primal, primal-dualGap, dualGap])
        end
        
        if stepSize === "agnostic"
            gamma = 2/(2+t)
        elseif stepSize === "linesearch"
           nothing, gamma = segmentSearch(f,grad,x,v,lsTol=min(max(eps(),dualGap ^ 2,epsilon ^ 2),epsilon ^ (1/2))) 
        elseif stepSize === "backtracking"
           nothing, gamma = backtrackingLS(f,grad,x,v,lsTol=min(max(eps(),dualGap ^ 2,epsilon ^ 2),epsilon ^ (1/2))) 
        end
        
        x = (1-gamma) * x + gamma * v
        
        if mod(t,printIt) == 0 && !silent
            tt = "FW"
            if t === 0
                tt = "I"
            end
            rep = [tt, string(t), primal, primal-dualGap, dualGap, (time_ns() - timeEl)/1.0e9]
            # timeEl = time_ns()
            println(tp.row(rep,width=width,format_spec="5e"))
            flush(stdout)
        end
        t = t + 1
    end
    if !silent
        println(tp.bottom(length(width),width=width))
        flush(stdout)
    end
    return x, v, primal, dualGap, trajData
end





end
