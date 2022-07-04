using FrankWolfe
using ProgressMeter
using Arpack
using Plots
using DoubleFloats
using ReverseDiff

using LinearAlgebra
using Random
import GLPK
using JSON

n = 200
k = 3000
#k = 500

xpi = rand(n * n);
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
target_accuracy = 1e-7

# modify to GLPK variant
# o = GLPK.Optimizer()
# lmo_moi = FrankWolfe.convert_mathopt(lmo, o, dimension=n)
# x00 = FrankWolfe.compute_extreme_point(lmo, direction_vec)

FrankWolfe.benchmark_oracles(
    x -> cf(x, xp),
    (str, x) -> cgrad!(str, x, xp),
    () -> randn(n, n),
    lmo;
    k=100,
)


# vanllia FW

x0 = copy(x00)

x, v, primal, dual_gap, trajectoryFW = FrankWolfe.frank_wolfe(
    x -> cf(x, xp),
    (str, x) -> cgrad!(str, x, xp),
    lmo,
    x0,
    max_iteration=k,
    line_search=FrankWolfe.Adaptive(),
    print_iter=k / 10,
    epsilon=target_accuracy,
    memory_mode=FrankWolfe.InplaceEmphasis(),
    trajectory=true,
    verbose=true,
);


# arbitrary cache

x0 = copy(x00)

x, v, primal, dual_gap, trajectoryLCG = FrankWolfe.lazified_conditional_gradient(
    x -> cf(x, xp),
    (str, x) -> cgrad!(str, x, xp),
    lmo,
    x0,
    max_iteration=k,
    epsilon=target_accuracy,
    line_search=FrankWolfe.Adaptive(),
    print_iter=k / 10,
    memory_mode=FrankWolfe.InplaceEmphasis(),
    trajectory=true,
    verbose=true,
);


# fixed cache size

x0 = copy(x00)

x, v, primal, dual_gap, trajectoryBLCG = FrankWolfe.lazified_conditional_gradient(
    x -> cf(x, xp),
    (str, x) -> cgrad!(str, x, xp),
    lmo,
    x0,
    max_iteration=k,
    line_search=FrankWolfe.Adaptive(),
    print_iter=k / 10,
    epsilon=target_accuracy,
    memory_mode=FrankWolfe.InplaceEmphasis(),
    trajectory=true,
    cache_size=500,
    verbose=true,
);

# AFW run

x0 = copy(x00)

x, v, primal, dual_gap, trajectoryLAFW = FrankWolfe.away_frank_wolfe(
    x -> cf(x, xp),
    (str, x) -> cgrad!(str, x, xp),
    lmo,
    x0,
    max_iteration=k,
    line_search=FrankWolfe.Adaptive(),
    print_iter=k / 10,
    epsilon=target_accuracy,
    memory_mode=FrankWolfe.InplaceEmphasis(),
    lazy=true,
    trajectory=true,
    verbose=true,
);


# BCG run

x0 = copy(x00)

x, v, primal, dual_gap, trajectoryBCG = FrankWolfe.blended_conditional_gradient(
    x -> cf(x, xp),
    (str, x) -> cgrad!(str, x, xp),
    lmo,
    x0,
    max_iteration=k,
    line_search=FrankWolfe.Adaptive(),
    print_iter=k / 10,
    epsilon=target_accuracy,
    memory_mode=FrankWolfe.InplaceEmphasis(),
    trajectory=true,
    verbose=true,
);


# BCG run (reference optimum)

x0 = copy(x00)

x, v, primal, dual_gap, trajectoryBCG_ref = FrankWolfe.blended_conditional_gradient(
    x -> cf(x, xp),
    (str, x) -> cgrad!(str, x, xp),
    lmo,
    x0,
    max_iteration=2 * k,
    line_search=FrankWolfe.Adaptive(),
    print_iter=k / 10,
    epsilon=target_accuracy / 10.0,
    memory_mode=FrankWolfe.InplaceEmphasis(),
    trajectory=true,
    verbose=true,
);

open("lcg_expensive_data.json", "w") do f
    return write(
        f,
        JSON.json((
            FW=trajectoryFW,
            LCG=trajectoryLCG,
            BLCG=trajectoryBLCG,
            LAFW=trajectoryLAFW,
            BCG=trajectoryBCG,
            reference_BCG_primal=primal,
        )),
    )
end

data = [trajectoryFW, trajectoryLCG, trajectoryBLCG, trajectoryLAFW, trajectoryBCG]
label = ["FW", "L-CG", "BL-CG", "L-AFW", "BCG"]
plot_trajectories(data, label, xscalelog=true)
