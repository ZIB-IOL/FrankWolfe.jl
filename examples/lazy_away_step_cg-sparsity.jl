import FrankWolfe
using LinearAlgebra
using Random

n = Int(1e4)
k = 1000

s = rand(1:100)
@info "Seed $s"
Random.seed!(s)

xpi = rand(n);
total = sum(xpi);
const xp = xpi ./ total;

f(x) = norm(x - xp)^2
function grad!(storage, x)
    @. storage = 2 * (x - xp)
end

const lmo = FrankWolfe.KSparseLMO(100, 1.0)

# const lmo_big = FrankWolfe.KSparseLMO(100, big"1.0")
# const lmo = FrankWolfe.LpNormLMO{Float64,5}(1.0)
# lmo = FrankWolfe.ProbabilitySimplexOracle(1.0);
# lmo = FrankWolfe.UnitSimplexOracle(1.0);

# 5 lpnorm issue with zero gradient 
const x00 = FrankWolfe.compute_extreme_point(lmo, rand(n))

# const lmo = FrankWolfe.BirkhoffPolytopeLMO()
# cost = rand(n, n)
# const x00 = FrankWolfe.compute_extreme_point(lmo, cost)



FrankWolfe.benchmark_oracles(f, grad!, () -> randn(n), lmo; k=100)

x0 = deepcopy(x00)
@time x, v, primal, dual_gap, trajectorySs = FrankWolfe.frank_wolfe(
    f,
    grad!,
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
@time x, v, primal, dual_gap, trajectoryAda = FrankWolfe.away_frank_wolfe(
    f,
    grad!,
    lmo,
    x0,
    max_iteration=k,
    line_search=FrankWolfe.adaptive,
    print_iter=k / 10,
    emphasis=FrankWolfe.memory,
    verbose=true,
    trajectory=true,
);


println("\n==> Lazy AFW.\n")

x0 = deepcopy(x00)
@time x, v, primal, dual_gap, trajectoryAdaLoc = FrankWolfe.away_frank_wolfe(
    f,
    grad!,
    lmo,
    x0,
    max_iteration=k,
    line_search=FrankWolfe.adaptive,
    print_iter=k / 10,
    emphasis=FrankWolfe.memory,
    verbose=true,
    lazy=true,
    K=1.5,
    trajectory=true,
);


x0 = deepcopy(x00)
@time x, v, primal, dual_gap, trajectoryAdaLoc5 = FrankWolfe.away_frank_wolfe(
    f,
    grad!,
    lmo,
    x0,
    max_iteration=k,
    line_search=FrankWolfe.adaptive,
    print_iter=k / 10,
    emphasis=FrankWolfe.memory,
    verbose=true,
    lazy=true,
    K=2.0,
    trajectory=true,
);

x0 = deepcopy(x00)
@time x, v, primal, dual_gap, trajectoryAdaLoc25 = FrankWolfe.away_frank_wolfe(
    f,
    grad!,
    lmo,
    x0,
    max_iteration=k,
    line_search=FrankWolfe.adaptive,
    print_iter=k / 10,
    emphasis=FrankWolfe.memory,
    verbose=true,
    K=4.0,
    lazy=true,
    trajectory=true,
);

x0 = deepcopy(x00)
@time x, v, primal, dual_gap, trajectoryAdaLoc1 = FrankWolfe.away_frank_wolfe(
    f,
    grad!,
    lmo,
    x0,
    max_iteration=k,
    line_search=FrankWolfe.adaptive,
    print_iter=k / 10,
    emphasis=FrankWolfe.memory,
    lazy=true,
    K=10.0,
    verbose=true,
    trajectory=true,
);


# data = [trajectorySs, trajectoryAda, trajectoryBCG]
# label = ["short step", "AFW", "BCG"]

data = [trajectorySs, trajectoryAda, trajectoryAdaLoc]
label = ["short step", "AFW", "AFW-Loc"]

# FrankWolfe.plot_trajectories(data, label, filename="convergence.pdf")

dataSparsity =
    [trajectoryAda, trajectoryAdaLoc, trajectoryAdaLoc5, trajectoryAdaLoc25, trajectoryAdaLoc1]
labelSparsity = ["AFW", "LAFW-K066", "LAFW-K05", "LAFW-K025", "LAFW-K01"]

FrankWolfe.plot_sparsity(dataSparsity, labelSparsity, filename="sparse.pdf")
