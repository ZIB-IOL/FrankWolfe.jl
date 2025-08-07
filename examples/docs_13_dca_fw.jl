# # Difference-of-Convex Algorithm with Frank-Wolfe
#
# This example shows the optimization of a difference-of-convex problem of the form:
# ```math
# min_{x \in \mathcal{X}} \phi(x) = f(x) - g(x)
# ```
# with $f$, $g$ convex functions with access to subgradients and $f$ smooth.
#
# The DCA-FW algorithm constructs local convex models of $\phi$ by linearizing $g$ and approximately optimizes them with FW.

using FrankWolfe
using LinearAlgebra
using Random
using SparseArrays
using StableRNGs
using Random

# The convex functions will be generated as random convex quadratics.
# minimize φ(x) = f(x) - g(x) where:
# f(x) = 0.5 * x^T A x + a^T x + c  (convex quadratic)
# g(x) = 0.5 * x^T B x + b^T x + d  (convex quadratic)

# ## Setting up the problem functions and data

const n = 500  # Reduced dimension

# Generate random positive definite matrices to ensure convexity
function generate_problem_data(rng)
    A_raw = randn(rng, n, n)
    A_raw ./= opnorm(A_raw)
    A = A_raw' * A_raw + 0.1 * I

    B_raw = randn(rng, n, n)
    B_raw ./= opnorm(B_raw)
    B = B_raw' * B_raw + 0.1 * I

    a = randn(rng, n)
    b = randn(rng, n)

    c = randn(rng)
    d = -359434.0

    return A, B, a, b, c, d
end

const rng = StableRNGs.StableRNG(1)
const A, B, a, b, c, d = generate_problem_data(rng)

function f(x)
    return 0.5 * FrankWolfe.fast_dot(x, A, x) + dot(a, x) + c
end

function grad_f!(storage, x)
    mul!(storage, A, x)
    storage .+= a
    return nothing
end

function g(x)
    return 0.5 * FrankWolfe.fast_dot(x, B, x) + dot(b, x) + d
end

function grad_g!(storage, x)
    mul!(storage, B, x)
    storage .+= b
    return nothing
end

# True objective function for verification
function phi(x)
    return f(x) - g(x)
end

lmo = FrankWolfe.KSparseLMO(5, 1000.0)

x0 = FrankWolfe.compute_extreme_point(lmo, randn(n))

x_final, primal_final, traj_data, dca_gap_final, iterations = FrankWolfe.dca_fw(
    f,
    grad_f!,
    g,
    grad_g!,
    lmo,
    copy(x0),
    max_iteration=200, # Outer iterations
    max_inner_iteration=10000, # Inner iterations
    epsilon=1e-5, # Tolerance for DCA gap
    line_search=FrankWolfe.Secant(),
    verbose=true,
    trajectory=true,
    verbose_inner=false,
    print_iter=10,
    use_corrective_fw=true,
    use_dca_early_stopping=true,
    grad_f_workspace=collect(x0),
    grad_g_workspace=collect(x0),
)

_, _, traj_data_boosted, _, _ = FrankWolfe.dca_fw(
    f,
    grad_f!,
    g,
    grad_g!,
    lmo,
    copy(x0),
    boosted=true,
    max_iteration=200, # Outer iterations
    max_inner_iteration=10000, # Inner iterations
    epsilon=1e-5, # Tolerance for DCA gap
    line_search=FrankWolfe.Secant(),
    verbose=true,
    trajectory=true,
    verbose_inner=false,
    print_iter=10,
    use_corrective_fw=true,
    use_dca_early_stopping=true,
    grad_f_workspace=collect(x0),
    grad_g_workspace=collect(x0),
)


# ## Plotting the resulting trajectory

data = [traj_data, traj_data_boosted]
label = ["DCA-FW", "DCA-FW-B"]
p_res = plot_trajectories(data, label, marker_shapes=[:o, :x])
ylabel!(p_res.subplots[3], "DCA gap")

