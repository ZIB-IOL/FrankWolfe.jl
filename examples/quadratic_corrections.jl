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
n_samples = 1000
K = 50
τ = 1.0  # right-hand side for K-sparse polytope
max_iter = 5000
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
    storage .-= linear_term
end

lmo = FrankWolfe.KSparseLMO(K, τ)
x0 = FrankWolfe.compute_extreme_point(lmo, zeros(n_features))

L = opnorm(hessian, 2)
line_search = FrankWolfe.Adaptive(L_est=L)

common_kw = (;
    max_iteration=max_iter,
    epsilon=target_tolerance,
    line_search=line_search,
    verbose=true,
    trajectory=true,
)

# --- Blended Pairwise CG (baseline) ---
@info "Blended Pairwise CG"
result_bpcg = FrankWolfe.blended_pairwise_conditional_gradient(
    f, grad!, lmo, copy(x0); common_kw...,
)
traj_bpcg = result_bpcg.traj_data
@info "  primal=$(result_bpcg.primal) dual_gap=$(result_bpcg.dual_gap) status=$(result_bpcg.status)"

# --- Quadratic LS correction (no MNP) ---
@info "Quadratic LS correction (MNP off)"
as_ls_no_mnp = FrankWolfe.ActiveSetQuadraticProductCaching([(1.0, copy(x0))], hessian, linear_term)
step_ls_no_mnp = FrankWolfe.QuadraticLSCorrection(hessian, linear_term, false)
result_qc_ls_no_mnp = FrankWolfe.corrective_frank_wolfe(
    f, grad!, lmo, step_ls_no_mnp, as_ls_no_mnp; common_kw...
)
traj_qc_ls_no_mnp = result_qc_ls_no_mnp.traj_data
@info "  primal=$(result_qc_ls_no_mnp.primal) dual_gap=$(result_qc_ls_no_mnp.dual_gap) status=$(result_qc_ls_no_mnp.status)"

# --- Quadratic LS correction (MNP on) ---
@info "Quadratic LS correction (MNP on)"
as_ls_mnp = FrankWolfe.ActiveSetQuadraticProductCaching([(1.0, copy(x0))], hessian, linear_term)
step_ls_mnp = FrankWolfe.QuadraticLSCorrection(hessian, linear_term, true)
result_qc_ls_mnp = FrankWolfe.corrective_frank_wolfe(
    f, grad!, lmo, step_ls_mnp, as_ls_mnp; common_kw...
)
traj_qc_ls_mnp = result_qc_ls_mnp.traj_data
@info "  primal=$(result_qc_ls_mnp.primal) dual_gap=$(result_qc_ls_mnp.dual_gap) status=$(result_qc_ls_mnp.status)"

# --- Quadratic LP correction (no MNP) ---
@info "Quadratic LP correction (MNP off)"
as_lp_no_mnp = FrankWolfe.ActiveSetQuadraticProductCaching([(1.0, copy(x0))], hessian, linear_term)
step_lp_no_mnp = FrankWolfe.QuadraticLPCorrection(hessian, linear_term, false)
result_qc_lp_no_mnp = FrankWolfe.corrective_frank_wolfe(
    f, grad!, lmo, step_lp_no_mnp, as_lp_no_mnp; common_kw..., traj_data=[]
)
traj_qc_lp_no_mnp = result_qc_lp_no_mnp.traj_data
@info "  primal=$(result_qc_lp_no_mnp.primal) dual_gap=$(result_qc_lp_no_mnp.dual_gap) status=$(result_qc_lp_no_mnp.status)"

# --- Quadratic LP correction (MNP on) ---
@info "Quadratic LP correction (MNP on)"
as_lp_mnp = FrankWolfe.ActiveSetQuadraticProductCaching([(1.0, copy(x0))], hessian, linear_term)
step_lp_mnp = FrankWolfe.QuadraticLPCorrection(hessian, linear_term, true)
result_qc_lp_mnp = FrankWolfe.corrective_frank_wolfe(
    f, grad!, lmo, step_lp_mnp, as_lp_mnp; common_kw..., traj_data=[]
)
traj_qc_lp_mnp = result_qc_lp_mnp.traj_data
@info "  primal=$(result_qc_lp_mnp.primal) dual_gap=$(result_qc_lp_mnp.dual_gap) status=$(result_qc_lp_mnp.status)"

# Plot comparison
data = [traj_bpcg, traj_qc_ls_no_mnp, traj_qc_ls_mnp, traj_qc_lp_no_mnp, traj_qc_lp_mnp]
labels = [
    "BlendedPairwise CG",
    "QC-LS (MNP off)",
    "QC-LS (MNP on)",
    "QC-LP (MNP off)",
    "QC-LP (MNP on)",
]
plot_trajectories(data, labels; xscalelog=true)
