import FrankWolfe
using LinearAlgebra
using Random
import GLPK

n = Int(1e2)
k = 3000

xpi = rand(n*n);
total = sum(xpi);
# next line needs to be commented out if we use the GLPK variants
xpi = reshape(xpi, n, n)
const xp = xpi # ./ total;

# better for memory consumption as we do coordinate-wise ops

function cf(x, xp)
    return LinearAlgebra.norm(x .- xp)^2 / n^2 
end

function cgrad!(storage, x, xp)
    return @. storage = 2 * (x - xp) / n^2 
end

# initial direction for first vertex
direction_vec = Vector{Float64}(undef, n * n)
randn!(direction_vec)
direction_mat = reshape(direction_vec, n, n)

lmo = FrankWolfe.BirkhoffPolytopeLMO()
x00 = FrankWolfe.compute_extreme_point(lmo, direction_mat)

# modify to GLPK variant
# o = GLPK.Optimizer()
# lmo = FrankWolfe.convert_mathopt(lmo, o, dimension=n)
# x00 = FrankWolfe.compute_extreme_point(lmo, direction_vec)

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
# TODO/Question: does not work with sparse structure as the memory allocation is not clear?

# x0 = deepcopy(x00)

# @time x, v, primal, dual_gap, trajectoryBLCG = FrankWolfe.lcg(
#     x -> cf(x, xp),
#     (str, x) -> cgrad!(str, x, xp),
#     lmo,
#     x0,
#     max_iteration=k,
#     L=100,
#     line_search=FrankWolfe.adaptive,
#     print_iter=k / 10,
#     emphasis=FrankWolfe.memory,
#     trajectory=true,
#     cache_size=500,
#     verbose=true,
# );


# BCG run

x0 = deepcopy(x00)

@time x, v, primal, dual_gap, trajectoryBCG = FrankWolfe.bcg(
    x -> cf(x, xp),
    (str, x) -> cgrad!(str, x, xp),
    lmo,
    x0,
    max_iteration=k,
    L=100,
    line_search=FrankWolfe.adaptive,
    print_iter=k / 10,
    linesearch_tol = 1e-9,
    emphasis=FrankWolfe.memory,
    trajectory=true,
    verbose=true,
);


data = [trajectoryFW, trajectoryLCG, trajectoryBCG]
label = ["FW" "LCG" "BCG"]

FrankWolfe.plot_trajectories(data, label)
