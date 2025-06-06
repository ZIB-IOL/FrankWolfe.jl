import FrankWolfe
using LinearAlgebra
using Random
using Test
using SparseArrays
using StableRNGs

n = Int(1e4)
k = 1000

s = 41
Random.seed!(StableRNG(s), s)

xpi = rand(n);
total = sum(xpi);
const xp = xpi # ./ total;

f(x) = norm(x .- xp)^2
function grad!(storage, x)
    @. storage = 2 * (x .- xp)
end

lmo = FrankWolfe.KSparseLMO(100, 1.0)

x0 = FrankWolfe.compute_extreme_point(lmo, spzeros(size(xp)...))
gradient = similar(xp)

x, v, primal, dual_gap, _, _ = FrankWolfe.blended_conditional_gradient(
    f,
    grad!,
    lmo,
    x0,
    max_iteration=k,
    line_search=FrankWolfe.AdaptiveZerothOrder(L_est=2.0),
    print_iter=100,
    memory_mode=FrankWolfe.InplaceEmphasis(),
    verbose=false,
    trajectory=false,
    sparsity_control=1.0,
    weight_purge_threshold=1e-9,
    epsilon=1e-8,
    gradient=gradient,
)

@test dual_gap ≤ 1e-3
@test f(x0) - f(x) ≥ 180

x0 = FrankWolfe.compute_extreme_point(lmo, spzeros(size(xp)...))

x, v, primal_cut, dual_gap, _, _ = FrankWolfe.blended_conditional_gradient(
    f,
    grad!,
    lmo,
    x0,
    max_iteration=k,
    line_search=FrankWolfe.AdaptiveZerothOrder(L_est=2.0),
    print_iter=k / 10,
    memory_mode=FrankWolfe.InplaceEmphasis(),
    verbose=false,
    trajectory=false,
    sparsity_control=1.0,
    weight_purge_threshold=1e-10,
    epsilon=1e-9,
    timeout=3.0,
)

@test primal ≤ primal_cut
