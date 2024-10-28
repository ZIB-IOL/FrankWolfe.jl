#=
This example demonstrates the use of the Blended Pairwise Conditional Gradient algorithm
with direct solve steps for a quadratic optimization problem over a sparse polytope.

The example showcases how the algorithm balances between:
- Pairwise steps for efficient optimization
- Periodic direct solves for handling the quadratic objective
- Lazy (approximate) linear minimization steps for improved iteration complexity

It also demonstrates how to set up custom callbacks for tracking algorithm progress.
=#

using FrankWolfe
using LinearAlgebra
using Random

import HiGHS

include("../examples/plot_utils.jl")

n = Int(1e4)
k = 10000

s = 10
@info "Seed $s"
Random.seed!(s)

xpi = rand(n);
total = sum(xpi);

# here the optimal solution lies in the interior if you want an optimal solution on a face and not the interior use:
# const xp = xpi;

const xp = xpi ./ total;

f(x) = norm(x - xp)^2
function grad!(storage, x)
    @. storage = 2 * (x - xp)
end

# lmo = FrankWolfe.KSparseLMO(2, 1.0)

## other LMOs to try
# lmo_big = FrankWolfe.KSparseLMO(100, big"1.0")
lmo = FrankWolfe.LpNormLMO{Float64,5}(1.0)
# lmo = FrankWolfe.ProbabilitySimplexOracle(1.0);
# lmo = FrankWolfe.UnitSimplexOracle(1.0);

x00 = FrankWolfe.compute_extreme_point(lmo, rand(n))

function build_callback(trajectory_arr)
    return function callback(state, active_set, args...)
        return push!(trajectory_arr, (FrankWolfe.callback_state(state)..., length(active_set)))
    end
end

FrankWolfe.benchmark_oracles(f, grad!, () -> randn(n), lmo; k=100)

trajectoryBPCG_standard = []
@time x, v, primal, dual_gap, _ = FrankWolfe.blended_pairwise_conditional_gradient(
    f,
    grad!,
    lmo,
    copy(x00),
    max_iteration=k,
    line_search=FrankWolfe.Shortstep(2.0),
    verbose=true,
    trajectory=true,
    callback=build_callback(trajectoryBPCG_standard),
);

trajectoryBPCG_as_sparse = []
active_set_sparse = FrankWolfe.ActiveSetSparsifier(
    FrankWolfe.ActiveSet([1.0], [x00], similar(x00)),
    HiGHS.Optimizer(),
)
@time x, v, primal, dual_gap, _ = FrankWolfe.blended_pairwise_conditional_gradient(
    f,
    grad!,
    lmo,
    active_set_sparse,
    max_iteration=k,
    line_search=FrankWolfe.Shortstep(2.0),
    verbose=true,
    callback=build_callback(trajectoryBPCG_as_sparse),
);

# Reduction primal/dual error vs. sparsity of solution

plot_data = [trajectoryBPCG_standard, trajectoryBPCG_as_sparse]
plot_labels = ["BPCG (Standard)", "BPCG (Sparsify)"]

# Plot sparsity
plot_sparsity(plot_data, plot_labels, legend_position=:topright)
