
# simple backtracking line search (not optimized)
# TODO:
# - code needs optimization

function backtrackingLS(f,grad,x,y;stepSize=true,lsTol=1e-10,stepLim=20,lsTau = 0.5)
    gamma = 1
    d = y - x
    i = 0
    gradDirection = dot(grad(x),d)
    
    if gradDirection === 0
        return i, 0
    end
        
    oldVal = f(x)
    newVal = f(x + gamma * d)
    while newVal - oldVal > lsTol * gamma * gradDirection
        if i > stepLim
            if oldVal - newVal >= 0
                return i, gamma
            else
                return i, 0
            end
        end
        gamma = gamma * lsTau
        newVal = f(x + gamma * d)
        i = i + 1
    end
    return i, gamma
end

# simple golden-ratio based line search (not optimized)
# based on boostedFW paper code and adapted for julia
# TODO:
# - code needs optimization 

function segmentSearch(f,grad,x,y;stepSize=true,lsTol=1e-10)
    # restrict segment of search to [x, y]
    d = (y-x) 
    left, right = copy(x), copy(y)

    # if the minimum is at an endpoint
    if dot(d, grad(x)) * dot(d, grad(y)) >= 0
        if f(y) <= f(x)
            return y, 1
        else
            return x, 0
        end
    end
    
    # apply golden-section method to segment
    gold = (1.0+sqrt(5)) / 2.0
    improv = Inf
    while improv > lsTol
        old_left, old_right = left, right
        new = left + (right - left) / (1.0+gold)
        probe = new + (right - new) / 2.0 
        if f(probe) <= f(new)
            left, right = new, right
        else
            left, right = left, probe
        end
        improv = norm(f(right) - f(old_right)) + norm(f(left)-f(old_left))
    end
    
    x_min = (left + right) / 2.0

    # compute step size gamma
    gamma = 0
    if stepSize === true
        for i in 1:length(d)
            if d[i] != 0
                gamma = (x_min[i]-x[i])/d[i]
                break
            end
        end
    end

    return x_min, gamma
end

