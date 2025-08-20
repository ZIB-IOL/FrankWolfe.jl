#=
This example demonstrates the use of the Blended Pairwise Conditional Gradient algorithm
with direct solve steps for a quadratic optimization problem over a sparse polytope.

The example showcases how the algorithm balances between:
- Pairwise steps for efficient optimization
- Periodic direct solves for handling the quadratic objective
- Lazy (approximate) linear minimization steps for improved iteration complexity

It also demonstrates how to set up custom callbacks for tracking algorithm progress.
=#

using LinearAlgebra
using Test
using Random
using StableRNGs

using FrankWolfe
import HiGHS
import MathOptInterface as MOI

n = Int(1e4)
k = 10000

s = 10
rng = StableRNG(s)
Random.seed!(rng, s)

xpi = rand(rng, n);
total = sum(xpi);

# here the optimal solution lies in the interior if you want an optimal solution on a face and not the interior use:
# const xp = xpi;

const xp = xpi ./ total;

f(x) = norm(x - xp)^2
function grad!(storage, x)
    @. storage = 2 * (x - xp)
end

lmo = FrankWolfe.LpNormLMO{Float64,5}(1.0)

x000 = FrankWolfe.compute_extreme_point(lmo, rand(rng, n))

function build_callback(trajectory_arr)
    return function callback(state, active_set, args...)
        return push!(trajectory_arr, (FrankWolfe.callback_state(state)..., length(active_set)))
    end
end

trajectoryBPCG_standard = []
x, v, primal, dual_gap, status, _, _ = FrankWolfe.blended_pairwise_conditional_gradient(
    f,
    grad!,
    lmo,
    copy(x000),
    max_iteration=k,
    line_search=FrankWolfe.Shortstep(2.0),
    verbose=false,
    trajectory=false,
    callback=build_callback(trajectoryBPCG_standard),
);

o = HiGHS.Optimizer()
MOI.set(o, MOI.Silent(), true)
active_set_sparse =
    FrankWolfe.ActiveSetSparsifier(FrankWolfe.ActiveSet([1.0], [x000], similar(x000)), o)
trajectoryBPCG_as_sparse = []
x, v, primal, dual_gap, status, _, _ = FrankWolfe.blended_pairwise_conditional_gradient(
    f,
    grad!,
    lmo,
    copy(x000),
    max_iteration=k,
    line_search=FrankWolfe.Shortstep(2.0),
    verbose=false,
    callback=build_callback(trajectoryBPCG_as_sparse),
);

as_cardinality_bpcg = getindex.(trajectoryBPCG_standard, 6)
as_cardinality_sparse = getindex.(trajectoryBPCG_as_sparse, 6)
@test maximum(as_cardinality_sparse - as_cardinality_bpcg) <= 0

dual_gap_bpcg = getindex.(trajectoryBPCG_standard, 4)
dual_gap_sparse = getindex.(trajectoryBPCG_as_sparse, 4)
@test maximum(dual_gap_sparse - dual_gap_bpcg) <= 1e-7
