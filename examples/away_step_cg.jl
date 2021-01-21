import FrankWolfe
import LinearAlgebra


# n = Int(1e1)
n = Int(1e2)
k = 10000

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
    maxIt=k,
    step_size=FrankWolfe.adaptive,
    L=100,
    printIt=k / 10,
    emph=FrankWolfe.memory,
    verbose=true,
    trajectory=true
);

@time x, v, primal, dualGap, trajectoryA, active_set = FrankWolfe.afw(
    f,
    grad,
    lmo,
    x0,
    maxIt=k,
    step_size=FrankWolfe.adaptive,
    L=100,
    printIt=k / 10,
    emph=FrankWolfe.memory,
    verbose=true,
    awaySteps=true,
    trajectory=true
);

data = [trajectory, trajectoryA]
label = ["FW" "AFW"]

FrankWolfe.plot_trajectories(data, label)

