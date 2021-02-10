import FrankWolfe
using LinearAlgebra
using Random

n = Int(5e5)
k = 5000

s = rand(1:100)
@info "Seed $s"
Random.seed!(s)

xpi = rand(n);
total = sum(xpi);
const xp = xpi ./ total;

f(x) = norm(x - xp)^2
grad(x) = 2 * (x - xp)

# const lmo = FrankWolfe.KSparseLMO(100, 1.0)

# const lmo_big = FrankWolfe.KSparseLMO(100, big"1.0")
const lmo = FrankWolfe.LpNormLMO{Float64,5}(1.0)
# lmo = FrankWolfe.ProbabilitySimplexOracle(1.0);
# lmo = FrankWolfe.UnitSimplexOracle(1.0);

# 5 lpnorm issue with zero gradient 
const x00 = FrankWolfe.compute_extreme_point(lmo, rand(n))

# const lmo = FrankWolfe.BirkhoffPolytopeLMO()
# cost = rand(n, n)
# const x00 = FrankWolfe.compute_extreme_point(lmo, cost)



FrankWolfe.benchmark_oracles(f, grad, lmo, n; k=100, T=eltype(x00))

x0 = deepcopy(x00)
@time x, v, primal, dual_gap, trajectorySs = FrankWolfe.fw(
    f,
    grad,
    lmo,
    x0,
    max_iteration=k,
    line_search=FrankWolfe.shortstep,
    L=2,
    print_iter=k / 10,
    emphasis=FrankWolfe.memory,
    verbose=true,
    trajectory=true,
);

x0 = deepcopy(x00)
@time x, v, primal, dual_gap, trajectoryAda = FrankWolfe.afw(
f,
    grad,
    lmo,
    x0,
    max_iteration=k,
    line_search=FrankWolfe.adaptive,
    L=100,
    print_iter=k / 10,
    emphasis=FrankWolfe.memory,
    verbose=true,
    trajectory=true,
);


println("\n==> Localized AFW.\n")

# x0 = deepcopy(x00)
# x, v, primal, dual_gap, trajectoryBCG = FrankWolfe.bcg(
#     f,
#     grad,
#     lmo,
#     x0,
#     max_iteration=k,
#     line_search=FrankWolfe.backtracking,
#     print_iter=k / 10,
#     emphasis=FrankWolfe.memory,
#     L=2,
#     verbose=true,
#     trajectory=true,
#     Ktolerance=1.00,
#     goodstep_tolerance=0.95,
#     weight_purge_threshold=1e-10,
# )

x0 = deepcopy(x00)
@time x, v, primal, dual_gap, trajectoryAdaLoc = FrankWolfe.afw(
    f,
    grad,
    lmo,
    x0,
    max_iteration=k,
    localized=true,
    localizedFactor=0.66,
    line_search=FrankWolfe.adaptive,
    L=100,
    print_iter=k / 10,
    emphasis=FrankWolfe.memory,
    verbose=true,
    trajectory=true,
);



# data = [trajectorySs, trajectoryAda, trajectoryBCG]
# label = ["short step", "AFW", "BCG"]

data = [trajectorySs, trajectoryAda, trajectoryAdaLoc]
label = ["short step", "AFW", "AFW-Loc"]

FrankWolfe.plot_trajectories(data, label)

dataSparsity = [trajectoryAda, trajectoryAdaLoc]
labelSparsity = ["AFW", "AFW-Loc"]

FrankWolfe.plot_sparsity(dataSparsity, labelSparsity)
# FrankWolfe.plot_trajectories(data[2:2], label[2:2])

# using Plots
# plot(getindex.(trajectoryAda, 4), xaxis=:log, yaxis=:log)


# vs = getindex.(trajectoryBCG, 3)

# plot(vs)
