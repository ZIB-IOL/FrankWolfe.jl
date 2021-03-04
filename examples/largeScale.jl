import FrankWolfe
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

FrankWolfe.benchmark_oracles(x -> cf(x, xp), (str, x) -> cgrad!(str, x, xp), () -> randn(n), lmo; k=100)

@time x, v, primal, dual_gap, trajectory = FrankWolfe.lcg(
    x -> cf(x, xp),
    (str, x) -> cgrad!(str, x, xp),
    lmo,
    x0,
    lazy=true,
    max_iteration=k,
    line_search=FrankWolfe.agnostic,
    print_iter=k / 10,
    emphasis=FrankWolfe.memory,
    verbose=true,
);
