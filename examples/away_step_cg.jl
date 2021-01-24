import FrankWolfe
import LinearAlgebra


# n = Int(1e1)
n = Int(1e2)
k = Int(1e4)

xpi = rand(n);
total = sum(xpi);
const xp = xpi # ./ total;

f(x) = LinearAlgebra.norm(x - xp)^2
grad(x) = 2 * (x - xp)

# problem with active set updates and the ksparselmo
lmo = FrankWolfe.KSparseLMO(40, 1);
# lmo = FrankWolfe.ProbabilitySimplexOracle(1)
x0 = FrankWolfe.compute_extreme_point(lmo, zeros(n));

FrankWolfe.benchmark_oracles(f, grad, lmo, n; k=100, T=Float64)

@time x, v, primal, dualGap, trajectory = FrankWolfe.fw(
    f,
    grad,
    lmo,
    x0,
    max_iteration=k,
    line_search=FrankWolfe.adaptive,
    L=100,
    print_iter=k / 10,
    Emphasis=FrankWolfe.memory,
    verbose=true,
    epsilon=1e-5,
    trajectory=true,
);

@time x, v, primal, dualGap, trajectoryA, active_set = FrankWolfe.afw(
    f,
    grad,
    lmo,
    x0,
    max_iteration=k,
    line_search=FrankWolfe.adaptive,
    L=100,
    print_iter=k / 10,
    epsilon=1e-5,
    Emphasis=FrankWolfe.memory,
    verbose=true,
    awaySteps=true,
    trajectory=true,
);

@time x, v, primal, dualGap, trajectoryAM, active_set = FrankWolfe.afw(
    f,
    grad,
    lmo,
    x0,
    max_iteration=k,
    line_search=FrankWolfe.adaptive,
    L=100,
    print_iter=k / 10,
    epsilon=1e-5,
    momentum=0.9,
    Emphasis=FrankWolfe.blas,
    verbose=true,
    awaySteps=true,
    trajectory=true,
);

data = [trajectory, trajectoryA, trajectoryAM]
label = ["FW" "AFW" "MAFW"]

FrankWolfe.plot_trajectories(data, label)
