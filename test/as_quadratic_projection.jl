using FrankWolfe
using LinearAlgebra
using Random

import HiGHS
import MathOptInterface as MOI
using Test
using StableRNGs

n = Int(1e2)
k = 10000

s = 10
Random.seed!(StableRNG(s), s)

xpi = rand(n);
total = sum(xpi);

xp = xpi ./ total;

f(x) = norm(x - xp)^2
function grad!(storage, x)
    @. storage = 2 * (x - xp)
end

# lmo = FrankWolfe.KSparseLMO(5, 1000.0)

## other LMOs to try
# lmo_big = FrankWolfe.KSparseLMO(100, big"1.0")
# lmo = FrankWolfe.LpNormLMO{Float64,5}(100.0)
# lmo = FrankWolfe.ProbabilitySimplexOracle(100.0);
lmo = FrankWolfe.UnitSimplexOracle(10000.0);

x00 = FrankWolfe.compute_extreme_point(lmo, rand(n))

function build_callback(trajectory_arr)
    return function callback(state, active_set, args...)
        return push!(trajectory_arr, (FrankWolfe.callback_state(state)..., length(active_set)))
    end
end

trajectoryBPCG_standard = []
x, v, primal, dual_gap, _ = FrankWolfe.blended_pairwise_conditional_gradient(
    f,
    grad!,
    lmo,
    copy(x00),
    max_iteration=k,
    callback=build_callback(trajectoryBPCG_standard),
);

active_set_quadratic_automatic_standard = FrankWolfe.ActiveSetQuadraticLinearSolve(
    FrankWolfe.ActiveSet([(1.0, copy(x00))]),
    grad!,
    MOI.instantiate(MOI.OptimizerWithAttributes(HiGHS.Optimizer, MOI.Silent() => true)),
    scheduler=FrankWolfe.LogScheduler(start_time=10, scaling_factor=1),
)
trajectoryBPCG_quadratic_automatic_standard = []
x, v, primal, dual_gap, _ = FrankWolfe.blended_pairwise_conditional_gradient(
    f,
    grad!,
    lmo,
    active_set_quadratic_automatic_standard,
    max_iteration=k,
    callback=build_callback(trajectoryBPCG_quadratic_automatic_standard),
);

active_set_quadratic_automatic = FrankWolfe.ActiveSetQuadraticLinearSolve(
    [(1.0, copy(x00))],
    grad!,
    MOI.instantiate(MOI.OptimizerWithAttributes(HiGHS.Optimizer, MOI.Silent() => true)),
    scheduler=FrankWolfe.LogScheduler(start_time=10, scaling_factor=1),
)
trajectoryBPCG_quadratic_automatic = []
x, v, primal, dual_gap, _ = FrankWolfe.blended_pairwise_conditional_gradient(
    f,
    grad!,
    lmo,
    active_set_quadratic_automatic,
    max_iteration=k,
    callback=build_callback(trajectoryBPCG_quadratic_automatic),
);

active_set_quadratic_manual = FrankWolfe.ActiveSetQuadraticLinearSolve(
    FrankWolfe.ActiveSet([(1.0, copy(x00))]),
    2.0 * I, -2xp,
    MOI.instantiate(MOI.OptimizerWithAttributes(HiGHS.Optimizer, MOI.Silent() => true)),
    scheduler=FrankWolfe.LogScheduler(start_time=10, scaling_factor=1),
)
trajectoryBPCG_quadratic_manual = []
x, v, primal, dual_gap, _ = FrankWolfe.blended_pairwise_conditional_gradient(
    f,
    grad!,
    lmo,
    active_set_quadratic_manual,
    max_iteration=k,
    callback=build_callback(trajectoryBPCG_quadratic_manual),
);

traj_data = [
    trajectoryBPCG_standard,
    trajectoryBPCG_quadratic_automatic_standard,
    trajectoryBPCG_quadratic_automatic,
    trajectoryBPCG_quadratic_manual,
]

# all should have converged
for traj in traj_data
    @test traj[end][2] ≤ 1e-8
    @test traj[end][4] ≤ 1e-7
end

lmo = FrankWolfe.LpNormLMO{Float64,5}(100.0)

active_set_quadratic_manual_wolfe = FrankWolfe.ActiveSetQuadraticLinearSolve(
    FrankWolfe.ActiveSet([(1.0, copy(x00))]),
    2.0 * I, -2xp,
    MOI.instantiate(MOI.OptimizerWithAttributes(HiGHS.Optimizer, MOI.Silent() => true)),
    scheduler=FrankWolfe.LogScheduler(start_time=10, scaling_factor=1),
    wolfe_step=true,
)

trajectoryBPCG_quadratic_manual = []
x, v, primal, dual_gap, _ = FrankWolfe.blended_pairwise_conditional_gradient(
    f,
    grad!,
    lmo,
    active_set_quadratic_manual,
    max_iteration=k,
    callback=build_callback(trajectoryBPCG_quadratic_manual),
);
trajectoryBPCG_quadratic_manual_wolfe = []
x, v, primal, dual_gap, _ = FrankWolfe.blended_pairwise_conditional_gradient(
    f,
    grad!,
    lmo,
    active_set_quadratic_manual_wolfe,
    max_iteration=k,
    callback=build_callback(trajectoryBPCG_quadratic_manual),
);

@test length(trajectoryBPCG_quadratic_manual) < 450
@test length(trajectoryBPCG_quadratic_manual_wolfe) < 450