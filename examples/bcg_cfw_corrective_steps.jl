# Compare Blended Conditional Gradients (BCG) with Corrective Frank-Wolfe
# using the three hull steps: SiGD, projected GD, and accelerated PGD.
#
# Feasible set: convex hull of randomly generated vertices.
# Objective: least-squares linear regression.

using FrankWolfe
using LinearAlgebra
using Random

include(joinpath(@__DIR__, "plot_utils.jl"))

Random.seed!(42)

n = 100
n_samples = 500
n_vertices = 100
max_iteration = 10000
epsilon = 1e-7
print_iter = max_iteration ÷ 10

vertices = [randn(n) for _ in 1:n_vertices]
lmo = FrankWolfe.ConvexHullLMO(vertices)

# Ground-truth parameter in the hull, noisy linear measurements
w_true = rand(n_vertices)
w_true ./= sum(w_true)
θ_true = sum(w_true[i] * vertices[i] for i in 1:n_vertices)

X = randn(n_samples, n)
y = X * θ_true + 0.1 * randn(n_samples)

hessian = Symmetric(X' * X)
Xt_y = X' * y
L = eigmax(hessian)

f(θ) = 0.5 * norm(X * θ - y)^2
function grad!(storage, θ)
    mul!(storage, hessian, θ)
    storage .-= Xt_y
    return storage
end

x00 = FrankWolfe.compute_extreme_point(lmo, randn(n))
common_kwargs = (
    epsilon=epsilon,
    max_iteration=max_iteration,
    print_iter=print_iter,
    line_search=FrankWolfe.Secant(),
    verbose=true,
    trajectory=true,
    memory_mode=FrankWolfe.InplaceEmphasis(),
)

res_bcg = FrankWolfe.blended_conditional_gradient(f, grad!, lmo, copy(x00); common_kwargs...)

res_sigd = FrankWolfe.corrective_frank_wolfe(
    f,
    grad!,
    lmo,
    FrankWolfe.SimplexGradientDescentStep(true),
    FrankWolfe.ActiveSet([(1.0, copy(x00))]);
    common_kwargs...,
)

res_pgd = FrankWolfe.corrective_frank_wolfe(
    f,
    grad!,
    lmo,
    FrankWolfe.ProjectedGradientDescentStep(hessian=hessian, lazy=true),
    FrankWolfe.ActiveSet([(1.0, copy(x00))]);
    common_kwargs...,
)

res_agd = FrankWolfe.corrective_frank_wolfe(
    f,
    grad!,
    lmo,
    FrankWolfe.ProjectedGradientDescentStep(hessian=hessian, lazy=true, accelerated=true),
    FrankWolfe.ActiveSet([(1.0, copy(x00))]);
    common_kwargs...,
)

results = (res_bcg, res_sigd, res_pgd, res_agd)
labels = ["BCG", "CFW-SiGD", "CFW-PGD", "CFW-AGD"]

data = [res.traj_data for res in results]
plot_trajectories(data, labels, xscalelog=true)
