mutable struct TrackingGradient{G} <: Function
    grad!::G
    counter::Int
end

TrackingGradient(grad!) = TrackingGradient(grad!, 0)

function (tg::TrackingGradient)(storage, x)
    tg.counter += 1
    return tg.grad!(storage, x)
end


mutable struct TrackingObjective{F} <: Function
    f::F
    counter::Int
end

TrackingObjective(f) = TrackingObjective(f, 0)

function (tf::TrackingObjective)(x)
    tf.counter += 1
    return tf.f(x)
end

mutable struct TrackingLMO{LMO} <: Function
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

function build_tracking_callback(f_values, dual_values, function_calls, gradient_calls, lmo_calls, time_vec)
    function tracking_callback(state)
        push!(function_calls, state.f.counter)
        push!(gradient_calls, state.grad!.counter)
        push!(gradient_calls, state.lmo_calls.counter)
        push!(f_values, state.primal)
        push!(f_values, state.primal)
        push!(dual_values, state.dual_gap)
        push!(lmo_calls, state.lmo.counter)
        push!(time_vec, state.time)
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

