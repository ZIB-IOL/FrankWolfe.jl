using FrankWolfe
using ProgressMeter
using Arpack
using Plots

using LinearAlgebra

n = Int(1e4)
k = 10000

xpi = rand(n);
total = sum(xpi);
const xp = xpi # ./ total;

f(x) = dot(x, x) - 2 * dot(x, xp)
function grad!(storage, x)
    @. storage = 2 * (x - xp)
end

# lmo = FrankWolfe.ProbabilitySimplexOracle(1);

lmo = FrankWolfe.KSparseLMO(100, 1.0)
x00 = FrankWolfe.compute_extreme_point(lmo, zeros(n));

# arbitrary cache

x0 = deepcopy(x00)

@time x, v, primal, dual_gap, trajectory = FrankWolfe.lazified_conditional_gradient(
    f,
    grad!,
    lmo,
    x0,
    max_iteration=k,
    line_search=FrankWolfe.Adaptive(L_est=100.0),
    print_iter=k / 10,
    memory_mode=FrankWolfe.InplaceEmphasis(),
    verbose=true,
);


# fixed cache size

x0 = deepcopy(x00)

@time x, v, primal, dual_gap, trajectory = FrankWolfe.lazified_conditional_gradient(
    f,
    grad!,
    lmo,
    x0,
    max_iteration=k,
    line_search=FrankWolfe.Adaptive(L_est=100.0),
    print_iter=k / 10,
    memory_mode=FrankWolfe.InplaceEmphasis(),
    cache_size=500,
    verbose=true,
);
