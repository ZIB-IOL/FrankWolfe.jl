import FrankWolfe
import LinearAlgebra
using LinearAlgebra

# n = Int(1e1)
n = Int(1e4)
k = Int(1e4)
number_nonzero = 40

xpi = rand(n);
total = sum(xpi);
const xp = xpi # ./ total;

f(x) = norm(x - xp)^2
function grad!(storage, x)
    @. storage = 2 * (x - xp)
end

# problem with active set updates and the ksparselmo
lmo = FrankWolfe.KSparseLMO(number_nonzero, 1);
# lmo = FrankWolfe.ProbabilitySimplexOracle(1)
x0 = FrankWolfe.compute_extreme_point(lmo, ones(n));

@time x, v, primal, dual_gap, trajectorylazy, active_set = FrankWolfe.lazy_afw(
    f,
    grad!,
    lmo,
    x0,
    max_iteration=k,
    line_search=FrankWolfe.adaptive,
    L=100,
    print_iter=k / 10,
    epsilon=1e-5,
    emphasis=FrankWolfe.memory,
    verbose=true,
    trajectory=true,
    lazy = true,
);

@time x, v, primal, dual_gap, trajectoryAFW, active_set = FrankWolfe.lazy_afw(
    f,
    grad!,
    lmo,
    x0,
    max_iteration=k,
    line_search=FrankWolfe.adaptive,
    L=100,
    print_iter=k / 10,
    epsilon=1e-5,
    emphasis=FrankWolfe.memory,
    verbose=true,
    awaySteps=true,
    trajectory=true,
);

@time x, v, primal, dual_gap, trajectoryFW = FrankWolfe.lazy_afw(
    f,
    grad!,
    lmo,
    x0,
    max_iteration=k,
    line_search=FrankWolfe.adaptive,
    L=100,
    print_iter=k / 10,
    emphasis=FrankWolfe.memory,
    verbose=true,
    epsilon=1e-5,
    trajectory=true,
    awaySteps = false,
);

@time x, v, primal, dual_gap, trajectoryLAFW = FrankWolfe.lazy_afw(
    f,
    grad!,
    lmo,
    x0,
    max_iteration=k,
    line_search=FrankWolfe.adaptive,
    L=100,
    print_iter=k / 10,
    emphasis=FrankWolfe.memory,
    verbose=true,
    epsilon=1e-5,
    trajectory=true,
    localized = true,
);







data = [trajectory, trajectoryA, trajectoryAM]
label = ["FW" "AFW" "MAFW"]

FrankWolfe.plot_trajectories(data, label)
