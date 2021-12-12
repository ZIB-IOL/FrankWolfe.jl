
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

function (tf::TrackingObjective)(x)
    tf.counter += 1
    return tf.f(x)
end

function compute_extreme_point(lmo::TrackingLMO, x; kwargs...)
    lmo.counter += 1
    return compute_extreme_point(lmo.inner, x; kwargs...)
end

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
