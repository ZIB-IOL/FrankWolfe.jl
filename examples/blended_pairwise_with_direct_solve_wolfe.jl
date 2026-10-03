#=
This example demonstrates the use of the Blended Pairwise Conditional Gradient algorithm
with direct solve steps for a quadratic optimization problem over a sparse polytope which is not standard quadratic.

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
import MathOptInterface as MOI

include("../examples/plot_utils.jl")

n = Int(1e2)
k = 10000

# s = rand(1:100)
s = 10
@info "Seed $s"
Random.seed!(s)

A = let
    A = randn(n, n)
    A' * A
end
@assert isposdef(A) == true

const y = Random.rand(Bool, n) * 0.6 .+ 0.3

function f(x)
    d = x - y
    return dot(d, A, d)
end

function grad!(storage, x)
    mul!(storage, A, x)
    return mul!(storage, A, y, -2, 2)
end


xpi = rand(n);
total = sum(xpi);

xp = xpi ./ total;

f(x) = norm(x - xp)^2
function grad!(storage, x)
    @. storage = 2 * (x - xp)
end

lmo = FrankWolfe.KSparseLMO(5, 500.0)

## other LMOs to try
lmo = FrankWolfe.KSparseLMO(10, big"500.0")
# lmo = FrankWolfe.LpNormBallLMO{Float64,5}(100.0)
# lmo = FrankWolfe.ProbabilitySimplexLMO(100.0);
# lmo = FrankWolfe.UnitSimplexLMO(10000.0);

x00 = FrankWolfe.compute_extreme_point(lmo, rand(n))


function build_callback(trajectory_arr)
    return function callback(state, active_set, args...)
        return push!(trajectory_arr, (FrankWolfe.callback_state(state)..., length(active_set)))
    end
end


trajectoryBPCG_standard = []
x, v, primal, dual_gap, status, _ = FrankWolfe.blended_pairwise_conditional_gradient(
    f,
    grad!,
    lmo,
    copy(x00),
    max_iteration=k,
    verbose=true,
    callback=build_callback(trajectoryBPCG_standard),
);


active_set_quadratic_automatic_standard = FrankWolfe.ActiveSetQuadraticLinearSolve(
    FrankWolfe.ActiveSet([(1.0, copy(x00))]),
    grad!,
    MOI.instantiate(MOI.OptimizerWithAttributes(HiGHS.Optimizer, MOI.Silent() => true)),
)
trajectoryBPCG_quadratic_automatic_standard = []
x, v, primal, dual_gap, status, _ = FrankWolfe.blended_pairwise_conditional_gradient(
    f,
    grad!,
    lmo,
    active_set_quadratic_automatic_standard,
    max_iteration=k,
    verbose=true,
    callback=build_callback(trajectoryBPCG_quadratic_automatic_standard),
);


active_set_quadratic_wolfe = FrankWolfe.ActiveSetQuadraticLinearSolve(
    FrankWolfe.ActiveSet([(1.0, copy(x00))]),
    2I,
    -2xp,
    MOI.instantiate(MOI.OptimizerWithAttributes(HiGHS.Optimizer, MOI.Silent() => true)),
    scheduler=FrankWolfe.LogScheduler(start_time=10, scaling_factor=1),
    wolfe_step=true,
)
trajectoryBPCG_quadratic_wolfe = []
x, v, primal, dual_gap, status, _ = FrankWolfe.blended_pairwise_conditional_gradient(
    f,
    grad!,
    lmo,
    active_set_quadratic_wolfe,
    max_iteration=k,
    verbose=true,
    callback=build_callback(trajectoryBPCG_quadratic_wolfe),
);

dataSparsity = [
    trajectoryBPCG_standard,
    trajectoryBPCG_quadratic_automatic_standard,
    trajectoryBPCG_quadratic_wolfe,
]
labelSparsity = ["BPCG (Standard)", "AS_Standard", "AS_Wolfe"]

# Plot trajectories
plot_trajectories(dataSparsity, labelSparsity, xscalelog=false)

# plot_sparsity(dataSparsity, labelSparsity, xscalelog=false)
