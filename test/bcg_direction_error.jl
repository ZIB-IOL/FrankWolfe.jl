import FrankWolfe
using LinearAlgebra
using Random
using Test
using SparseArrays

n = Int(1e4)
k = 3000

s = 41
Random.seed!(s)

xpi = rand(n);
total = sum(xpi);
xp = xpi # ./ total;

f(x) = norm(x - xp)^2
function grad!(storage, x)
    @. storage = 2 * (x - xp)
end

lmo = FrankWolfe.KSparseLMO(100, 1.0)

x0 = FrankWolfe.compute_extreme_point(lmo, spzeros(size(xp)...))

x, v, primal, dual_gap, _ = FrankWolfe.blended_conditional_gradient(
    f,
    grad!,
    lmo,
    x0,
    max_iteration=k,
    line_search=FrankWolfe.Adaptive(),
    print_iter=k / 10,
    emphasis=FrankWolfe.memory,
    L=2,
    verbose=true,
    trajectory=false,
    lazy_tolerance=1.0,
    weight_purge_threshold=1e-10,
    epsilon=1e-9,
)

@test dual_gap ≤ 5e-4
@test f(x0) - f(x) ≥ 180

x0 = FrankWolfe.compute_extreme_point(lmo, spzeros(size(xp)...))

x, v, primal_cut, dual_gap, _ = FrankWolfe.blended_conditional_gradient(
    f,
    grad!,
    lmo,
    x0,
    max_iteration=k,
    line_search=FrankWolfe.Adaptive(),
    print_iter=k / 10,
    emphasis=FrankWolfe.memory,
    L=2,
    verbose=true,
    trajectory=false,
    lazy_tolerance=1.0,
    weight_purge_threshold=1e-10,
    epsilon=1e-9,
    timeout=3.0,
)

@test primal ≤ primal_cut
