import FrankWolfe
using LinearAlgebra
using Random

n = Int(3e2)
k = 1000

xpi = rand(n*n);
total = sum(xpi);
xpi = reshape(xpi, n, n)
const xp = xpi # ./ total;

# better for memory consumption as we do coordinate-wise ops

function cf(x, xp)
    return LinearAlgebra.norm(x .- xp)^2
end

function cgrad!(storage, x, xp)
    return @. storage = 2 * (x - xp)
end

# initial direction for first vertex
direction_vec = Vector{Float64}(undef, n * n)
randn!(direction_vec)
direction_mat = reshape(direction_vec, n, n)

lmo = FrankWolfe.BirkhoffPolytopeLMO()
x00 = FrankWolfe.compute_extreme_point(lmo, direction_mat)

FrankWolfe.benchmark_oracles(x -> cf(x, xp), (str, x) -> cgrad!(str, x, xp), () -> randn(n,n), lmo; k=100)


# vanllia FW

x0 = deepcopy(x00)

@time x, v, primal, dual_gap, trajectoryFW = FrankWolfe.fw(
    x -> cf(x, xp),
    (str, x) -> cgrad!(str, x, xp),
    lmo,
    x0,
    max_iteration=k,
    L=100,
    line_search=FrankWolfe.adaptive,
    print_iter=k / 10,
    emphasis=FrankWolfe.memory,
    trajectory=true,
    verbose=true,
);


# arbitrary cache

x0 = deepcopy(x00)

@time x, v, primal, dual_gap, trajectoryLCG = FrankWolfe.lcg(
    x -> cf(x, xp),
    (str, x) -> cgrad!(str, x, xp),
    lmo,
    x0,
    max_iteration=k,
    L=100,
    line_search=FrankWolfe.adaptive,
    print_iter=k / 10,
    emphasis=FrankWolfe.memory,
    trajectory=true,
    verbose=true,
);


# fixed cache size

x0 = deepcopy(x00)

@time x, v, primal, dual_gap, trajectoryBLCG = FrankWolfe.lcg(
    x -> cf(x, xp),
    (str, x) -> cgrad!(str, x, xp),
    lmo,
    x0,
    max_iteration=k,
    L=100,
    line_search=FrankWolfe.adaptive,
    print_iter=k / 10,
    emphasis=FrankWolfe.memory,
    trajectory=true,
    cache_size=500,
    verbose=true,
);


data = [trajectoryFW, trajectoryLCG, trajectoryFW]
label = ["FW" "LCG" "BLCG"]

FrankWolfe.plot_trajectories(data, label)
