"""
A function acting like the normal `grad!`
but tracking the number of calls.
"""
mutable struct TrackingGradient{G} <: Function
    grad!::G
    counter::Int
end

TrackingGradient(grad!) = TrackingGradient(grad!, 0)

function (tg::TrackingGradient)(storage, x)
    tg.counter += 1
    return tg.grad!(storage, x)
end


"""
A function acting like the normal objective `f`
but tracking the number of calls.
"""
mutable struct TrackingObjective{F} <: Function
    f::F
    counter::Int
end

TrackingObjective(f) = TrackingObjective(f, 0)

function (tf::TrackingObjective)(x)
    tf.counter += 1
    return tf.f(x)
end

function wrap_objective(to::TrackingObjective)
    function f(x)
        to.counter += 1
        return to.f(x)
    end
    function grad!(storage,x)
        to.counter += 1
        return to.g(storage ,x)
    end
    return (f,grad!)
end

"""
    TrackingLMO{LMO}(lmo)

An LMO wrapping another one and tracking the number of calls.
"""
mutable struct TrackingLMO{LMO} <: LinearMinimizationOracle
    lmo::LMO
    counter::Int
end

function compute_extreme_point(lmo::TrackingLMO, x; kwargs...)
    lmo.counter += 1
    return compute_extreme_point(lmo.lmo, x)
end

is_tracking_lmo(lmo) = false
is_tracking_lmo(lmo::TrackingLMO) = true

TrackingLMO(lmo) = TrackingLMO(lmo, 0)
