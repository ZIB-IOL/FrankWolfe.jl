using FrankWolfe
using ProgressMeter
using Arpack
using Plots
using DoubleFloats
using ReverseDiff

using LinearAlgebra
using Random
import GLPK

s = rand(1:100)
s = 98
@info "Seed $s"
Random.seed!(s)


n = Int(1e2)
k = 3000

xpi = rand(n * n);
total = sum(xpi);
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

# BirkhoffPolytopeLMO via Hungarian Method
lmo_native = FrankWolfe.BirkhoffPolytopeLMO()

# BirkhoffPolytopeLMO realized via LP solver
lmo_moi = FrankWolfe.convert_mathopt(lmo_native, GLPK.Optimizer(), dimension=n)

# choose between lmo_native (= Hungarian Method) and lmo_moi (= LP formulation solved with GLPK)
lmo = lmo_native

x00 = FrankWolfe.compute_extreme_point(lmo, direction_mat)

FrankWolfe.benchmark_oracles(
    x -> cf(x, xp),
    (str, x) -> cgrad!(str, x, xp),
    () -> randn(n, n),
    lmo;
    k=100,
)


# BCG run

x0 = deepcopy(x00)

@time x, v, primal, dual_gap, trajectoryBCG = FrankWolfe.blended_conditional_gradient(
    x -> cf(x, xp),
    (str, x) -> cgrad!(str, x, xp),
    lmo,
    x0,
    max_iteration=k,
    line_search=FrankWolfe.Adaptive(L_est=100.0),
    print_iter=k / 10,
    memory_mode=FrankWolfe.InplaceEmphasis(),
    trajectory=true,
    verbose=true,
);


data = [trajectoryBCG]
label = ["BCG"]

plot_trajectories(data, label)
