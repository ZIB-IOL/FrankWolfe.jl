#=
Linear regression over the K-sparse polytope: compare
- Blended Pairwise CG (baseline)
- Quadratic LS correction (linear solve) with MNP on/off
- Quadratic LP correction with MNP on/off

Objective: minimize (1/2)‖Xβ - y‖² over β in the K-sparse polytope.
Hessian A = X'X, linear term b = -X'y.
=#

using FrankWolfe
using LinearAlgebra
using Random

import HiGHS
import MathOptInterface as MOI

include(joinpath(@__DIR__, "plot_utils.jl"))

# Problem size
n_features = 500
n_samples = 5000
K = 50
τ = 1.0  # right-hand side for K-sparse polytope
max_iter = 10000
target_tolerance = 1e-6

Random.seed!(42)
X = randn(n_samples, n_features)
y = randn(n_samples)
# Normalize for better conditioning
X .= X ./ sqrt(n_samples)

# Linear regression: f(β) = (1/2)‖Xβ - y‖² = (1/2)β'X'Xβ - y'Xβ + const
# Hessian A = X'X, gradient = X'Xβ - X'y, so b = -X'y
hessian = X' * X
linear_term = -X' * y

f(β) = 0.5 * norm(X * β - y)^2
function grad!(storage, β)
    mul!(storage, X', X * β)
    storage .+= linear_term
end

lmo = FrankWolfe.KSparseLMO(K, τ)
x0 = FrankWolfe.compute_extreme_point(lmo, zeros(n_features))

common_kw = (;
    max_iteration=max_iter,
    epsilon=target_tolerance,
    verbose=true,
    trajectory=true,
)

# BPCG
result_bpcg = FrankWolfe.blended_pairwise_conditional_gradient(
    f, grad!, lmo, copy(x0); common_kw...,lazy=true,
)
traj_bpcg = result_bpcg.traj_data

scheduler = FrankWolfe.make_default_scheduler(2, 2.0, 1000)

# QC-LS
as_ls_no_mnp = FrankWolfe.ActiveSetQuadraticProductCaching([(1.0, copy(x0))], hessian, linear_term)
step_ls_no_mnp = FrankWolfe.ScheduledStep(
    FrankWolfe.BlendedPairwiseStep(true),           # BPCG-style fallback step
    FrankWolfe.QuadraticLSCorrection(hessian, linear_term, false),  # QC step
    scheduler,
)
result_qc_ls_no_mnp = FrankWolfe.corrective_frank_wolfe(
    f, grad!, lmo, step_ls_no_mnp, as_ls_no_mnp; common_kw...
)
traj_qc_ls_no_mnp = result_qc_ls_no_mnp.traj_data

# QC-LS MNP
as_ls_mnp = FrankWolfe.ActiveSetQuadraticProductCaching([(1.0, copy(x0))], hessian, linear_term)
step_ls_mnp = FrankWolfe.ScheduledStep(
    FrankWolfe.BlendedPairwiseStep(true),
    FrankWolfe.QuadraticLSCorrection(hessian, linear_term, true),
    scheduler,
)
result_qc_ls_mnp = FrankWolfe.corrective_frank_wolfe(
    f, grad!, lmo, step_ls_mnp, as_ls_mnp; common_kw...
)
traj_qc_ls_mnp = result_qc_ls_mnp.traj_data

# QC-LP
as_lp_no_mnp = FrankWolfe.ActiveSetQuadraticProductCaching([(1.0, copy(x0))], hessian, linear_term)
step_lp_no_mnp = FrankWolfe.ScheduledStep(
    FrankWolfe.BlendedPairwiseStep(true),
    FrankWolfe.QuadraticLPCorrection(
        hessian,
        linear_term,
        MOI.instantiate(MOI.OptimizerWithAttributes(HiGHS.Optimizer, MOI.Silent() => true)),
        false,
    ),
    scheduler,
)
result_qc_lp_no_mnp = FrankWolfe.corrective_frank_wolfe(
    f, grad!, lmo, step_lp_no_mnp, as_lp_no_mnp; common_kw..., traj_data=[]
)
traj_qc_lp_no_mnp = result_qc_lp_no_mnp.traj_data

# QC-LP MNP
as_lp_mnp = FrankWolfe.ActiveSetQuadraticProductCaching([(1.0, copy(x0))], hessian, linear_term)
step_lp_mnp = FrankWolfe.ScheduledStep(
    FrankWolfe.BlendedPairwiseStep(true),
    FrankWolfe.QuadraticLPCorrection(
        hessian,
        linear_term,
        MOI.instantiate(MOI.OptimizerWithAttributes(HiGHS.Optimizer, MOI.Silent() => true)),
        true,
    ),
    scheduler,
)
result_qc_lp_mnp = FrankWolfe.corrective_frank_wolfe(
    f, grad!, lmo, step_lp_mnp, as_lp_mnp; common_kw..., traj_data=[]
)
traj_qc_lp_mnp = result_qc_lp_mnp.traj_data

# Plot comparison
data = [traj_bpcg, traj_qc_ls_no_mnp, traj_qc_ls_mnp, traj_qc_lp_no_mnp, traj_qc_lp_mnp]
labels = [
    "BPCG",
    "QC-LS",
    "QC-LS MNP",
    "QC-LP",
    "QC-LP MNP",
]
plot_trajectories(data, labels; xscalelog=true)
