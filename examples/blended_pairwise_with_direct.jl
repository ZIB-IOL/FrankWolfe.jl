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

import Pkg
Pkg.add("GLPK")
Pkg.add("HiGHS")
Pkg.add("Clp")
import GLPK
import HiGHS
import Clp

lp_solver = GLPK.Optimizer
# lp_solver = HiGHS.Optimizer
# lp_solver = Clp.Optimizer # buggy / does not work properly

include("../examples/plot_utils.jl")

n = Int(1e4)
k = 10000

# s = rand(1:100)
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

lmo = FrankWolfe.KSparseLMO(2, 1.0)

## other LMOs to try
# lmo_big = FrankWolfe.KSparseLMO(100, big"1.0")
# lmo = FrankWolfe.LpNormLMO{Float64,5}(1.0)
# lmo = FrankWolfe.ProbabilitySimplexOracle(1.0);
# lmo = FrankWolfe.UnitSimplexOracle(1.0);

x00 = FrankWolfe.compute_extreme_point(lmo, rand(n))


## example with BirkhoffPolytopeLMO - uses square matrix.
# const lmo = FrankWolfe.BirkhoffPolytopeLMO()
# cost = rand(n, n)
# const x00 = FrankWolfe.compute_extreme_point(lmo, cost)


function build_callback(trajectory_arr)
    return function callback(state, active_set, args...)
        return push!(trajectory_arr, (FrankWolfe.callback_state(state)..., length(active_set)))
    end
end


FrankWolfe.benchmark_oracles(f, grad!, () -> randn(n), lmo; k=100)

trajectoryBPCG_standard = []
callback = build_callback(trajectoryBPCG_standard)

x0 = deepcopy(x00)
@time x, v, primal, dual_gap, _ = FrankWolfe.blended_pairwise_conditional_gradient(
    f,
    grad!,
    lmo,
    x0,
    max_iteration=k,
    line_search=FrankWolfe.Shortstep(2.0),
    print_iter=k / 10,
    memory_mode=FrankWolfe.InplaceEmphasis(),
    verbose=true,
    trajectory=true,
    callback=callback,
    squadratic=false,
);

trajectoryBPCG_quadratic = []
callback = build_callback(trajectoryBPCG_quadratic)

x0 = deepcopy(x00)
@time x, v, primal, dual_gap, _ = FrankWolfe.blended_pairwise_conditional_gradient(
    f,
    grad!,
    lmo,
    x0,
    max_iteration=k,
    line_search=FrankWolfe.Shortstep(2.0),
    print_iter=k / 10,
    memory_mode=FrankWolfe.InplaceEmphasis(),
    verbose=true,
    trajectory=true,
    callback=callback,
    squadratic=true,
    lp_solver=lp_solver,
);

# Reduction primal/dual error vs. sparsity of solution

dataSparsity = [trajectoryBPCG_standard, trajectoryBPCG_quadratic]
labelSparsity = ["BPCG (Standard)", "BPCG (Direct)"]

# Plot sparsity
# plot_sparsity(dataSparsity, labelSparsity, legend_position=:topright)

# Plot trajectories
plot_trajectories(dataSparsity, labelSparsity,xscalelog=false)
