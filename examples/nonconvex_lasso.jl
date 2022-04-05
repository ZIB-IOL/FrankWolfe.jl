
using FrankWolfe
using ProgressMeter
using Arpack
using Plots
using DoubleFloats
using ReverseDiff

using LinearAlgebra


n = Int(1e3);
k = 1e5

xpi = rand(n);
total = sum(xpi);
const xp = xpi ./ total;

f(x) = 2 * norm(x - xp)^3 - norm(x)^2

# this is just for the example -> better explicitly define your gradient
grad!(storage, x) = ReverseDiff.gradient!(storage, f, x)

# pick feasible region
lmo = FrankWolfe.ProbabilitySimplexOracle(1.0); #radius needs to be float

# compute some initial vertex
x0 = collect(FrankWolfe.compute_extreme_point(lmo, zeros(n)))

# benchmarking Oracles
FrankWolfe.benchmark_oracles(f, grad!, () -> randn(n), lmo; k=100)

@time x, v, primal, dual_gap, trajectory = FrankWolfe.frank_wolfe(
    f,
    grad!,
    lmo,
    x0,
    max_iteration=k,
    line_search=FrankWolfe.Nonconvex(),
    print_iter=k / 10,
    verbose=true,
);
