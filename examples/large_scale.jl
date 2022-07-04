#=

Running a very large-scale example with 1e9 variables (note this requires a lot of RAM).

Problem is quite simple: minimize || x - p ||^2 over the probability simplex

NOTE.
1. running standard FW with agnostic step-size here as overhead from line searches etc is quite substantial
2. observe that the memory consummption is sub-linear in the number of iterations

=#

using FrankWolfe
using ProgressMeter
using Arpack
using Plots
using DoubleFloats
using ReverseDiff

using LinearAlgebra

n = Int(1e7)
k = 1000

xpi = rand(n);
total = sum(xpi);
const xp = xpi ./ total;

f(x) = norm(x - xp)^2
function grad!(storage, x)
    @. storage = 2 * (x - xp)
end

# better for memory consumption as we do coordinate-wise ops

function cf(x, xp)
    return LinearAlgebra.norm(x .- xp)^2
end

function cgrad!(storage, x, xp)
    return @. storage = 2 * (x - xp)
end

lmo = FrankWolfe.ProbabilitySimplexOracle(1);
x0 = FrankWolfe.compute_extreme_point(lmo, zeros(n));

FrankWolfe.benchmark_oracles(
    x -> cf(x, xp),
    (str, x) -> cgrad!(str, x, xp),
    () -> randn(n),
    lmo;
    k=100,
)

@time x, v, primal, dual_gap, trajectory = FrankWolfe.frank_wolfe(
    x -> cf(x, xp),
    (str, x) -> cgrad!(str, x, xp),
    lmo,
    x0,
    max_iteration=k,
    line_search=FrankWolfe.Agnostic(),
    print_iter=k / 10,
    memory_mode=FrankWolfe.InplaceEmphasis(),
    verbose=true,
);
