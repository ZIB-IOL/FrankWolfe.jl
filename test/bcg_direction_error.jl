import FrankWolfe
using LinearAlgebra
using Random
using Test

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

x, v, primal, dual_gap, _ = FrankWolfe.bcg(
    f,
    grad!,
    lmo,
    x0,
    max_iteration=k,
    line_search=FrankWolfe.adaptive,
    print_iter=k / 10,
    emphasis=FrankWolfe.memory,
    L=2,
    verbose=false,
    trajectory=false,
    Ktolerance=1.00,
    goodstep_tolerance=0.95,
    weight_purge_threshold=1e-10,
    epsilon=1e-9,
)

@test dual_gap ≤ 5e-4
@test f(x0) - f(x) ≥ 180
