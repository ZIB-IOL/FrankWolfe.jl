include("activate.jl")

using LinearAlgebra

n = Int(1e4)
k = 10000

xpi = rand(n);
total = sum(xpi);
const xp = xpi # ./ total;

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

# lmo = FrankWolfe.ProbabilitySimplexOracle(1);

lmo = FrankWolfe.KSparseLMO(100, 1.0)
x00 = FrankWolfe.compute_extreme_point(lmo, zeros(n));

FrankWolfe.benchmark_oracles(
    x -> cf(x, xp),
    (str, x) -> cgrad!(str, x, xp),
    () -> randn(n),
    lmo;
    k=100,
)

# arbitrary cache

x0 = deepcopy(x00)

@time x, v, primal, dual_gap, trajectory = FrankWolfe.lazified_conditional_gradient(
    x -> cf(x, xp),
    (str, x) -> cgrad!(str, x, xp),
    lmo,
    x0,
    max_iteration=k,
    L=100,
    line_search=FrankWolfe.Adaptive(),
    print_iter=k / 10,
    emphasis=FrankWolfe.memory,
    verbose=true,
);


# fixed cache size

x0 = deepcopy(x00)

@time x, v, primal, dual_gap, trajectory = FrankWolfe.lazified_conditional_gradient(
    x -> cf(x, xp),
    (str, x) -> cgrad!(str, x, xp),
    lmo,
    x0,
    max_iteration=k,
    L=100,
    line_search=FrankWolfe.Adaptive(),
    print_iter=k / 10,
    emphasis=FrankWolfe.memory,
    cache_size=500,
    verbose=true,
);
