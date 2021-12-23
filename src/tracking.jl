"""
    TrackingGradient{G}
    A mutable struct that can act like the normal
    grad! function but with an additional method for count tracking
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
    TrackingObjective{F}
    A mutable struct that can act like the normal
    objective function f but with an additional method for count tracking
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

"""
    TrackingLMO{LMO}
    A mutable struct that can act like the normal
    LMO oracle but with an additional method for count tracking
"""
mutable struct TrackingLMO{LMO} <: LinearMinimizationOracle
    lmo::LMO
    counter::Int
end

"""
    compute_extreme_point(lmo::TrackingLMO, direction; kwargs...)
    Mirrors behaviour of same function found in oracles.jl, increasing
    call counter
Computes the point `argmin_{v ∈ C} v ⋅ direction`
with `C` the set represented by the LMO.
All LMOs should accept keyword arguments that they can ignore.
"""
function compute_extreme_point(lmo::TrackingLMO, x; kwargs...)
    lmo.counter += 1
    return compute_extreme_point(lmo.lmo, x)
end

is_tracking_lmo(lmo) = false
is_tracking_lmo(lmo::TrackingLMO) = true

TrackingLMO(lmo) = TrackingLMO(lmo, 0)

"""
    tracking_trajectory_callback(storage)
    similar to trajectory_callback found in utils.jl
Callback pushing the state at each iteration to the passed storage.
The state data is only the 5 first fields + 3 call counters usually:
`(t,primal,dual,dual_gap,time,function_calls,gradient_calls,lmo_calls)`
"""
function tracking_trajectory_callback(storage)
    return function tracking_push_trajectory!(state)
        return push!(storage, Tuple(state)[1:8])
    end
end

# function wrap_objective(to:TrackingObjective)
#     function f(x)
#         to.counter += 1
#         return to.f(x)
#     end
#     function grad!(storage,x)
#         to.counter += 1
#         return to.g(storage ,x)
#     end
#     return (f,grad!)
# end

