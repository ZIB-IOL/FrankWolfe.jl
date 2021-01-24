
include(joinpath(@__DIR__, "activate.jl"))

using LinearAlgebra


n = Int(1e3);
k = 1000

xpi = rand(n);
total = sum(xpi);
const xp = xpi ./ total;

const f(x) = 2 * LinearAlgebra.norm(x - xp)^3 - LinearAlgebra.norm(x)^2
const grad = x -> ReverseDiff.gradient(f, x) # this is just for the example -> better explicitly define your gradient

# pick feasible region
lmo = FrankWolfe.ProbabilitySimplexOracle(1.0); #radius needs to be float

# compute some initial vertex
x0 = FrankWolfe.compute_extreme_point(lmo, zeros(n));

# benchmarking Oracles
FrankWolfe.benchmark_oracles(f, grad, lmo, n; k=100, T=Float64)

@time x, v, primal, dualGap, trajectory = FrankWolfe.fw(
    f,
    grad,
    lmo,
    x0,
    max_iteration=k,
    line_search=FrankWolfe.nonconvex,
    print_iter=k / 10,
    verbose=true,
);
