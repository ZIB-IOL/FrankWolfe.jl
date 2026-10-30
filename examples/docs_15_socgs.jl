# # Second-order Conditional Gradient Sliding
#
# This example shows Second-order Conditional Gradient Sliding (SOCGS) on a
# smooth and strongly convex problem: ``\ell_2``-regularized logistic regression
# over the ``\ell_1`` ball,
# ```math
# \min_{\|x\|_1 \leq 1} f(x) = \frac{1}{m} \sum_{j=1}^m \log\left(1 + \exp(-y_j a_j^\top x)\right) + \frac{\lambda}{2} \|x\|_2^2.
# ```
#
# At an iterate ``x_k``, SOCGS builds the quadratic model
# ```math
# m_k(w) = f(x_k) + \langle \nabla f(x_k), w - x_k \rangle + \frac12 \langle H_k (w - x_k), w - x_k \rangle
# ```
# with the Hessian ``H_k = \nabla^2 f(x_k)`` and minimizes it approximately with a
# corrective Frank-Wolfe algorithm. Evaluating the model and its gradient requires
# no further gradient evaluations of ``f``. The subproblem is solved until its
# Frank-Wolfe gap is at most ``\tau \|z - x_k\|^4``, which suffices for local
# quadratic convergence. An independent conditional-gradient sequence on ``f``
# ensures global convergence; the next iterate is the best of ``x_k``, the
# subproblem solution, and the independent iterate.

using FrankWolfe
using LinearAlgebra
using Random
using StableRNGs

# ## Problem data

const n_features = 100
const n_samples = 1000
const λ = 0.05

rng = StableRNG(42)
const D = randn(rng, n_samples, n_features) / sqrt(n_features)
x_true = randn(rng, n_features)
const y = sign.(D * x_true .+ 0.1 * randn(rng, n_samples));

# The objective, its gradient, and its Hessian. With ``z_j = y_j a_j^\top x`` and
# the logistic function ``\sigma``, the gradient is
# ``\frac1m \sum_j -\sigma(-z_j) y_j a_j + \lambda x`` and the Hessian is
# ``\frac1m D^\top \operatorname{diag}(\sigma(z_j)(1 - \sigma(z_j))) D + \lambda I``,
# where the rows of the data matrix ``D`` are the feature vectors ``a_j``.

sigmoid(t) = 1 / (1 + exp(-t))
log1pexp(t) = max(t, zero(t)) + log1p(exp(-abs(t)))

function f(x)
    z = y .* (D * x)
    return sum(log1pexp, -z) / n_samples + λ / 2 * dot(x, x)
end

function grad!(storage, x)
    z = y .* (D * x)
    s = -y .* sigmoid.(-z) ./ n_samples
    mul!(storage, D', s)
    storage .+= λ .* x
    return nothing
end

function hessian!(H, x)
    z = y .* (D * x)
    w = sigmoid.(z) .* (1 .- sigmoid.(z)) ./ n_samples
    mul!(H, D', w .* D)
    H[diagind(H)] .+= λ
    return nothing
end

lmo = FrankWolfe.LpNormBallLMO{Float64,1}(1.0)
x0 = collect(FrankWolfe.compute_extreme_point(lmo, ones(n_features)));

# ## SOCGS
#
# The default inner solver of SOCGS is [`FrankWolfe.corrective_frank_wolfe`](@ref)
# with blended pairwise steps. The Hessian is passed as `hessian!(H, x)`, which
# fills the (by default dense) matrix `H`.

result_socgs = FrankWolfe.second_order_conditional_gradient_sliding(
    f,
    grad!,
    hessian!,
    lmo,
    copy(x0),
    tau=1.0,
    verbose=true,
);

# The `Type` column shows which point was selected: the subproblem solution
# (`PVM`), the independent conditional-gradient iterate (`CG`), or the previous
# iterate (`Keep`). The `Inner` column shows the number of iterations of the last
# subproblem.

# ## SOCGS with quadratic corrections
#
# Since the subproblems are quadratic, they can be accelerated with the quadratic
# correction steps of [Halbey, Rakotomandimby, Besançon, Designolle, Pokutta (2025)](https://arxiv.org/abs/2506.02635).
# These steps depend on the Hessian `A` and linear term `b` of the model, with
# ``\nabla m_k(w) = A w + b``. They are therefore passed as a builder
# `(A, b) -> CorrectiveStep`, which is called for every subproblem.

qc_builder =
    (A, b) -> FrankWolfe.ScheduledStep(
        FrankWolfe.BlendedPairwiseStep(true),
        FrankWolfe.QuadraticLSCorrection(A, b, true),
    )

result_socgs_qc = FrankWolfe.second_order_conditional_gradient_sliding(
    f,
    grad!,
    hessian!,
    lmo,
    copy(x0),
    inner_step=qc_builder,
    verbose=true,
);

# ## Comparison with BPCG
#
# We compare both variants with lazified blended pairwise conditional gradients
# (BPCG) applied directly to ``f``. To count LMO calls, we wrap the LMO in a
# [`FrankWolfe.TrackingLMO`](@ref) and record its counter in the trajectory. The
# methods are run without printing so that the running times are comparable.

function run_tracked(algorithm, args...; kwargs...)
    tracking_lmo = FrankWolfe.TrackingLMO(lmo)
    traj_data = []
    callback = function (state, args...)
        push!(traj_data, (tracking_lmo.counter, state.primal, state.dual, state.dual_gap, state.time))
        return true
    end
    algorithm(args..., tracking_lmo, copy(x0); callback=callback, kwargs...)
    return traj_data
end

bpcg = FrankWolfe.blended_pairwise_conditional_gradient
socgs = FrankWolfe.second_order_conditional_gradient_sliding
run_tracked(bpcg, f, grad!; lazy=true, max_iteration=10) # hide
run_tracked(socgs, f, grad!, hessian!; max_iteration=2) # hide
run_tracked(socgs, f, grad!, hessian!; max_iteration=2, inner_step=qc_builder) # hide
traj_bpcg = run_tracked(bpcg, f, grad!; lazy=true, max_iteration=100_000)
traj_socgs = run_tracked(socgs, f, grad!, hessian!)
traj_socgs_qc = run_tracked(socgs, f, grad!, hessian!; inner_step=qc_builder);

# We plot the primal value and the FW gap against the number of LMO calls and the
# running time. For SOCGS, each point corresponds to one outer iteration.

data = [traj_bpcg, traj_socgs, traj_socgs_qc]
labels = ["BPCG", "SOCGS", "SOCGS-QC"]
p_res = plot_grid(
    data,
    labels,
    [1, 5],
    [2, 4],
    ["LMO calls", "Time (s)"],
    ["Primal", "FW gap"],
    xscalelog=true,
    marker_shapes=[:o, :x, :+],
)

#
display(p_res)
