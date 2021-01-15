
"""
Supertype for linear minimization oracles.

All LMOs must implement `compute_extreme_point(lmo::LMO, direction)`
and return a vector `v` of the appropriate type.
"""
abstract type LinearMinimizationOracle
end

"""
    compute_extreme_point(lmo::LinearMinimizationOracle, direction; kwargs...)

Computes the point `argmin_{v ∈ C} v ⋅ direction`
with `C` the set represented by the LMO.
"""
function compute_extreme_point end



"""
lazified lmos
# TODO

"""

function lazy_compute_extreme_point_threshold(lmo::LinearMinimizationOracle, direction, threshold)
    tt::StepType
    if !isempty(lmo.cache)
        tt = lazylazy ## optimistically lazy -> reused last point
        v = lmo.v
        if dot(v, direction) > threshold # be optimistic: true last returned point first
            tt = lazy ## just lazy -> used point from cache
            test = (x -> dot(x, direction)).(lmo.cache) 
            v = lmo.cache[argmin(test)]
        end
        if dot(v, direction) > threshold  # still not good enough, then solve the LP
            tt = regular ## no cache point -> used the expensive LP
            v = compute_extreme_point(lmo, direction)
            if !(v in lmo.cache) 
                push!(lmo.cache,v)
            end
        end
    else    
        tt = regular
        v = compute_extreme_point(lmo, direction)
        push!(lmo.cache,v)
    end
    lmo.v = v
    return v, dot(v,direction), tt
end
