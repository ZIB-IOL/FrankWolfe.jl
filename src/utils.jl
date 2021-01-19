
"""
Slight modification of
Aaptive Step Size strategy from https://arxiv.org/pdf/1806.05123.pdf

TODO: 
- make emphasis aware and optimize
"""

function adaptive_step_size(f, gradient, x, v, L_est; eta = 0.9, tau = 2, gamma_max = 1)
    M = eta * L_est
    direction = x - v
    gamma = min(LinearAlgebra.dot(gradient, direction) / (M * LinearAlgebra.norm(direction)^2), gamma_max )
    while f(x - gamma * direction) - f(x) > - gamma * LinearAlgebra.dot(gradient, direction) + gamma^2 * M / 2.0 * LinearAlgebra.norm(direction)^2
        M *= tau
    end
    return M, gamma
end

# simple backtracking line search (not optimized)
# TODO:
# - code needs optimization

function backtrackingLS(f,grad,x,y;stepSize=true,lsTol=1e-10,stepLim=20,lsTau = 0.5)
    gamma = 1
    d = y - x
    i = 0
    gradDirection = LinearAlgebra.dot(grad(x),d)
    
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
    if LinearAlgebra.dot(d, grad(x)) * LinearAlgebra.dot(d, grad(y)) >= 0
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
        improv = LinearAlgebra.norm(f(right) - f(old_right)) + LinearAlgebra.norm(f(left)-f(old_left))
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

"""
    MaybeHotVector{T}

Represents a vector of at most one value different from 0.
"""
struct MaybeHotVector{T} <: AbstractVector{T}
    active_val::T
    val_idx::Int
    len::Int
end

Base.size(v::MaybeHotVector) = (v.len, )

@inline function Base.getindex(v::MaybeHotVector{T}, idx::Integer) where {T}
    @boundscheck if !( 1 ≤ idx ≤ length(v))
        throw(BoundsError(v, idx))
    end
    if v.val_idx != idx
        return zero(T)
    end
    return v.active_val
end

Base.sum(v::MaybeHotVector) = v.active_val

function LinearAlgebra.dot(v1::MaybeHotVector, v2::AbstractVector)
    return v1.active_val * v2[v1.val_idx]
end

LinearAlgebra.dot(v1::AbstractVector, v2::MaybeHotVector) = LinearAlgebra.dot(v2, v1)

# warning, no bound check
function LinearAlgebra.dot(v1::MaybeHotVector, v2::MaybeHotVector)
    if length(v1) != length(v2)
        throw(DimensionMismatch("v1 and v2 do not have matching sizes"))
    end
    return v1.active_val * v2.active_val * (v1.val_idx == v2.val_idx)
end

function Base.:*(v::MaybeHotVector, x::Number)
    MaybeHotVector(v.active_val * x, v.val_idx, v.len)
end

Base.:*(x::Number, v::MaybeHotVector) = v * x

##############################
### emphasis macro
##############################


macro emphasis(emph, ex)
    esc(quote
        if $emph === memory
            @. $ex
        else
            $ex
        end
    end)
    end
