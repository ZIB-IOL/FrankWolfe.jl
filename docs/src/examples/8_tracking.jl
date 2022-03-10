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

# The `FrankWolfe.TrackingCallback` stores in the trajectory array the
# number of calls to each oracle at each iteration.

x0 = FrankWolfe.compute_extreme_point(tlmo_prob, ones(5))
fw_results = FrankWolfe.frank_wolfe(
    tf,
    tgrad!,
    tlmo_prob,
    x0,
    max_iteration=1000,
    line_search=FrankWolfe.Agnostic(),
    callback=FrankWolfe.TrackingCallback(),
)

@show tf.counter
@show tgrad!.counter
@show tlmo_prob.counter;

# We can show the difference between this standard run and the
# lazified conditional gradient algorithm which avoids calling the LMO
# at some iterations.

tf.counter = 0
tgrad!.counter = 0
tlmo_prob.counter = 0

FrankWolfe.lazified_conditional_gradient(
    tf,
    tgrad!,
    tlmo_prob,
    x0,
    max_iteration=1000,
    line_search=FrankWolfe.Agnostic(),
    callback=FrankWolfe.TrackingCachedCallback(),
)

@show tf.counter
@show tgrad!.counter
@show tlmo_prob.counter;
