# Tracking options of the FrankWolfe.jl package

# In this example we will go through the process of building a vanilla FW setup with additional tracking functions
# Frankwolfe.jl is able to count function, gradient and LMO calls. 
# in order to track each of these metrics, a "Tracking" version of the Gradient, LMO and Function methods have to be supplied to 
# the frank_wolfe algorithm, which are constructed from their non-tracking counterparts. 

# We first import the necessary packages
using FrankWolfe
using Test
using LinearAlgebra

using DoubleFloats
using DelimitedFiles
import FrankWolfe: ActiveSet


# in order to count the number of function calls, a TrackingObjective function tf is defined right after the definition of the 
# objective function f
f(x) = norm(x)^2
tf = FrankWolfe.TrackingObjective(f,0)

#similarly, the tgrad! function tracks the number of gradient calls during the 
function grad!(storage, x)
    return storage .= 2x
end
tgrad! = FrankWolfe.TrackingGradient(grad!,0)

# the tracking function can be applied for all types of LMOs and even in a nasted way
lmo_prob = FrankWolfe.ProbabilitySimplexOracle(1)
tlmo_prob = FrankWolfe.TrackingLMO(lmo_prob)

x0 = FrankWolfe.compute_extreme_point(tlmo_prob, zeros(5))
# f_values, dual_values, function_calls, gradient_calls, lmo_calls, time_vec
tracking_trajectory_callback = FrankWolfe.tracking_trajectory_callback

fw_results = FrankWolfe.frank_wolfe(
        tf,
        tgrad!,
        tlmo_prob,
        x0,
        max_iteration=1000,
        line_search=FrankWolfe.Agnostic(),
        trajectory=true,
        callback=tracking_trajectory_callback,
        verbose=false,
)

#the trajectory will now contain for each iteration 3 additional cells detailing the counts of the objective function, gradient and LMO calls
x, v, primal, dual_gap, trajectory = fw_results


