# Tracking number of calls to different oracles

# In this example we will run the standard Frank-Wolfe algorithm while tracking the number of 
# calls to the different oracles, namely function, gradient evaluations, and LMO calls.
# In order to track each of these metrics, a "Tracking" version of the Gradient, LMO and Function methods have to be supplied to 
# the frank_wolfe algorithm, which are wrapping a standard one.

using FrankWolfe
using Test
using LinearAlgebra
import FrankWolfe: ActiveSet


# In order to count the number of function calls, a `TrackingObjective` is built from a standard objective function `f`.
f(x) = norm(x)^2
tf = FrankWolfe.TrackingObjective(f)

# Similarly, the `tgrad!` function tracks the number of gradient calls:

function grad!(storage, x)
    return storage .= 2x
end
tgrad! = FrankWolfe.TrackingGradient(grad!)

# The tracking function can be applied for all types of LMOs and even in a nested way, which can be useful to
# track the number of calls to a lazified oracle.

lmo_prob = FrankWolfe.ProbabilitySimplexOracle(1)
tlmo_prob = FrankWolfe.TrackingLMO(lmo_prob)

x0 = FrankWolfe.compute_extreme_point(tlmo_prob, ones(5))
fw_results = FrankWolfe.frank_wolfe(
    tf,
    tgrad!,
    tlmo_prob,
    x0,
    max_iteration=1000,
    line_search=FrankWolfe.Agnostic(),
    callback=nothing,
)

@show tf.counter
@show tgrad!.counter
@show tlmo_prob.counter;


# Adding our own callback


# We will 
# Let us first reset the counters of the tracking functions:
tf.counter = 0
tgrad!.counter = 0
tlmo_prob.counter = 0

# The `storage` variable stores in the trajectory array the
# number of calls to each oracle at each iteration.
storage = []

# Now define our own callback function to take advantage of the tracking functions
# we can extend the default trajectory logging by adding the ".counter" field arguments present in the tracking functions.
# (this function works for both vanilla and lazified frank_wolfe versions)
function push_tracking_state(state,storage)
    base_tuple = Tuple(state)[1:5]
    if typeof(state.lmo) <: FrankWolfe.CachedLinearMinimizationOracle
        complete_tuple = tuple(base_tuple..., state.gamma, state.f.counter, state.grad.counter, state.lmo.inner.counter)
    else
        complete_tuple = tuple(base_tuple..., state.gamma, state.f.counter, state.grad.counter, state.lmo.counter)
    end
    push!(storage, complete_tuple)
end

# In case we want to stop the frank_wolfe algorithm prematurely after a good enough value of the objective function f is reached,
# we can return a boolean stop criterion that will halt the optimization if the pimal objective function is evaluated more than 500 times.
function make_callback(storage)
    return function callback(state)
        if state.primal < Inf && isdefined(Base,:primal_limit) == false
            primal_limit = state.primal
        end
        push_tracking_state(state,storage)
        return state.f.counter > 500
    end
end

callback = make_callback(storage)

# We can show the difference between this standard run and the
# lazified conditional gradient algorithm which avoids calling the LMO
# at some iterations.

FrankWolfe.lazified_conditional_gradient(
    tf,
    tgrad!,
    tlmo_prob,
    x0,
    max_iteration=1000,
    traj_data = storage,
    line_search=FrankWolfe.Agnostic(),
    callback=callback,
)

@show storage[end][1]
@show tf.counter
@show tgrad!.counter
@show tlmo_prob.counter;