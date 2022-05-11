# # Tracking, counters and custom callbacks for Frank Wolfe

# In this example we will run the standard Frank-Wolfe algorithm while tracking the number of
# calls to the different oracles, namely function, gradient evaluations, and LMO calls.
# In order to track each of these metrics, a "Tracking" version of the Gradient, LMO and Function methods have to be supplied to
# the frank_wolfe algorithm, which are wrapping a standard one.

using FrankWolfe
using Test
using LinearAlgebra
using FrankWolfe: ActiveSet


# ## The trackers for primal objective, gradient and LMO.

# In order to count the number of function calls, a `TrackingObjective` is built from a standard objective function `f`,
# which will act in the same way as the original function does, but with an additional `.counter` field which tracks the number of calls.

f(x) = norm(x)^2
tf = FrankWolfe.TrackingObjective(f)
@show tf.counter
tf(rand(3))
@show tf.counter
## Resetting the counter
tf.counter = 0;

# Similarly, the `tgrad!` function tracks the number of gradient calls:

function grad!(storage, x)
    return storage .= 2x
end
tgrad! = FrankWolfe.TrackingGradient(grad!)
@show tgrad!.counter;

# The tracking LMO operates in a similar fashion and tracks the number of `compute_extreme_point` calls.

lmo_prob = FrankWolfe.ProbabilitySimplexOracle(1)
tlmo_prob = FrankWolfe.TrackingLMO(lmo_prob)
@show tlmo_prob.counter;

# The tracking LMO can be applied for all types of LMOs and even in a nested way, which can be useful to
# track the number of calls to a lazified oracle.
# We can now pass the tracking versions `tf`, `tgrad` and `tlmo_prob` to `frank_wolfe`
# and display their call counts after the optimization process.

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


# ## Adding a custom callback

# A callback is a user-defined function called at every iteration
# of the algorithm with the current state passed as a named tuple.
#
# We can implement our own callback, for example with:
# - Extended trajectory logging, similar to the `trajectory = true` option
# - Stop criterion after a certain number of calls to the primal objective function
#
# To reuse the same tracking functions, Let us first reset their counters:
tf.counter = 0
tgrad!.counter = 0
tlmo_prob.counter = 0;

# The `storage` variable stores in the trajectory array the
# number of calls to each oracle at each iteration.

storage = []

# Now define our own trajectory logging function that extends
# the five default logged elements `(iterations, primal, dual, dual_gap, time)` with ".counter" field arguments present in the tracking functions.

function push_tracking_state(state, storage)
    base_tuple = Tuple(state)[1:5]
    if typeof(state.lmo) <: FrankWolfe.CachedLinearMinimizationOracle
        complete_tuple = tuple(base_tuple..., state.gamma, state.f.counter, state.grad.counter, state.lmo.inner.counter)
    else
        complete_tuple = tuple(base_tuple..., state.gamma, state.f.counter, state.grad.counter, state.lmo.counter)
    end
    push!(storage, complete_tuple)
end

# In case we want to stop the frank_wolfe algorithm prematurely after a certain condition is met,
# we can return a boolean stop criterion `false`.
# Here, we will implement a callback that terminates the algorithm if the primal objective function is evaluated more than 500 times.
function make_callback(storage)
    return function callback(state)
        push_tracking_state(state,storage)
        return state.f.counter < 500
    end
end

callback = make_callback(storage)

# We can show the difference between this standard run and the
# lazified conditional gradient algorithm which does not call the LMO at each iteration.

FrankWolfe.lazified_conditional_gradient(
    tf,
    tgrad!,
    tlmo_prob,
    x0,
    max_iteration=1000,
    traj_data=storage,
    line_search=FrankWolfe.Agnostic(),
    callback=callback,
)

total_iterations = storage[end][1]
@show total_iterations
@show tf.counter
@show tgrad!.counter
@show tlmo_prob.counter;
