# # Quadratic corrections

# This example compares quadratic correction steps for a convex quadratic objective
# over the ``K``-sparse polytope
# ```math
# \mathcal{P}_{K,\tau} = \operatorname{conv}\bigl\{ x \in \{\pm \tau/K, 0\}^n : \|x\|_0 \le K \bigr\}.
# ```
# We solve a least-squares problem
# ```math
# \min_{x \in \mathcal{P}_{K,\tau}} \frac{1}{m}\|A x - y\|_2^2,
# ```
# so the Hessian is ``\nabla^2 f = (2/m) A^\top A`` and the linear term of
# ``\nabla f(x) = (2/m)(A^\top A x - A^\top y)`` is ``b = -(2/m) A^\top y``.

# The methods follow
# [Halbey, Rakotomandimby, Besançon, Designolle, Pokutta (2025)](https://arxiv.org/abs/2506.02635),
# *Efficient Quadratic Corrections for Frank-Wolfe Algorithms*:
#
# - **BPCG**: blended pairwise conditional gradients (local pairwise steps only).
# - **QC-LS**: [`FrankWolfe.QuadraticLSCorrection`](@ref) with `mnp=false` — affine
#   minimizer via a linear system, accepted only if it lies in the convex hull
#   (first branch of Algorithm 6).
# - **QC-LS MNP**: the same linear system with a Wolfe ratio test, i.e. **QC-MNP**
#   (Algorithm 6).
# - **QC-LP**: [`FrankWolfe.QuadraticLPCorrection`](@ref) with `mnp=false` —
#   **QC-LP** (Algorithm 5).
# - **QC-LP MNP**: the LP form of QC-MNP (Algorithm 8).
#
# Quadratic corrections are combined with BPCG pairwise steps through
# [`FrankWolfe.ScheduledStep`](@ref). If a non-MNP correction is infeasible, the
# pairwise step is used as fallback.

using FrankWolfe
using LinearAlgebra
using Random

import HiGHS
import MathOptInterface as MOI

include("plot_utils.jl")

# ## Problem data

n_features = 500
n_samples = 10000
K = 5
τ = 1.0
max_iter = 5000
target_tolerance = 1e-6

Random.seed!(1)
A = randn(n_samples, n_features)
y = randn(n_samples)
AA = A' * A
Aty = A' * y

hessian = (2 / n_samples) * AA
linear_term = -(2 / n_samples) * Aty

f(x) = (1 / n_samples) * norm(A * x - y)^2
function grad!(storage, x)
    mul!(storage, AA, x)
    storage .-= Aty
    storage .*= 2 / n_samples
end

lmo = FrankWolfe.KSparseLMO(K, τ)
x0 = FrankWolfe.compute_extreme_point(lmo, ones(n_features))

common_kw = (;
    max_iteration=max_iter,
    epsilon=target_tolerance,
    verbose=true,
    trajectory=true,
    print_iter=max_iter ÷ 5,
)

# ## Blended pairwise conditional gradients

result_bpcg = FrankWolfe.blended_pairwise_conditional_gradient(
    f,
    grad!,
    lmo,
    copy(x0);
    common_kw...,
    lazy=true,
)
traj_bpcg = result_bpcg.traj_data

# ## Quadratic corrections with a pairwise fallback
#
# All QC variants share the same scheduler and BPCG fallback. A new
# [`FrankWolfe.LogScheduler`](@ref) is created for each run so that the atom
# counter starts from zero.

qc_scheduler() = FrankWolfe.LogScheduler(start_time=10, scaling_factor=1.0)
bpcg_step() = FrankWolfe.BlendedPairwiseStep(true)
silent_optimizer() =
    MOI.instantiate(MOI.OptimizerWithAttributes(HiGHS.Optimizer, MOI.Silent() => true))

function run_qc(correction)
    active_set = FrankWolfe.ActiveSetQuadraticProductCaching(
        [(1.0, copy(x0))],
        hessian,
        linear_term,
    )
    step = FrankWolfe.ScheduledStep(bpcg_step(), correction, qc_scheduler())
    return FrankWolfe.corrective_frank_wolfe(f, grad!, lmo, step, active_set; common_kw...)
end

# QC-LS (linear system, no ratio test):

result_qc_ls = run_qc(FrankWolfe.QuadraticLSCorrection(hessian, linear_term, false))
traj_qc_ls = result_qc_ls.traj_data

# QC-LS MNP (Algorithm 6):

result_qc_ls_mnp = run_qc(FrankWolfe.QuadraticLSCorrection(hessian, linear_term, true))
traj_qc_ls_mnp = result_qc_ls_mnp.traj_data

# QC-LP (Algorithm 5):

result_qc_lp = run_qc(FrankWolfe.QuadraticLPCorrection(hessian, linear_term, silent_optimizer(), false))
traj_qc_lp = result_qc_lp.traj_data

# QC-LP MNP (Algorithm 8):

result_qc_lp_mnp = run_qc(FrankWolfe.QuadraticLPCorrection(hessian, linear_term, silent_optimizer(), true))
traj_qc_lp_mnp = result_qc_lp_mnp.traj_data

# ## Comparison
#
# QC-LS and QC-LP agree when the affine minimizer is unique (strongly convex
# quadratic). QC-MNP can drop atoms even when the affine minimizer lies outside
# the convex hull, so it typically makes more progress per correction than QC-LP,
# which then falls back to a pairwise step.

data = [traj_bpcg, traj_qc_ls, traj_qc_ls_mnp, traj_qc_lp, traj_qc_lp_mnp]
labels = ["BPCG", "QC-LS", "QC-LS MNP", "QC-LP", "QC-LP MNP"]
plot_trajectories(data, labels; xscalelog=false)
