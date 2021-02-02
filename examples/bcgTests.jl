import FrankWolfe
using LinearAlgebra
using Random

n = Int(1e4)
k = 5000

s = rand(1:100)
@info "Seed $s"
Random.seed!(s)

xpi = rand(n);
total = sum(xpi);
const xp = xpi ./ total;

f(x) = norm(x - xp)^2
grad(x) = 2 * (x - xp)

# better for memory consumption as we do coordinate-wise ops

function cf(x, xp)
    return @. norm(x - xp)^2
end

function cgrad(x, xp)
    return @. 2 * (x - xp)
end

lmo = FrankWolfe.KSparseLMO(100, 1.0)
# lmo = FrankWolfe.KSparseLMO(100, big"1.0")
# lmo = FrankWolfe.LpNormLMO{Float64,1}(1.0)
# lmo = FrankWolfe.ProbabilitySimplexOracle(1.0);
# lmo = FrankWolfe.UnitSimplexOracle(1.0);
const x00 = FrankWolfe.compute_extreme_point(lmo, zeros(BigFloat, n))
# print(x0)

FrankWolfe.benchmark_oracles(x -> cf(x, xp), x -> cgrad(x, xp), lmo, n; k=100, T=Float64)

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

# println("\n==> Goldenratio LS.\n")

# @time x, v, primal, dual_gap, trajectoryGr = FrankWolfe.fw(f,grad,lmo,x0,max_iteration=k,
#     line_search=FrankWolfe.goldenratio,L=100,print_iter=k/10,Emphasis=FrankWolfe.memory,verbose=true, trajectory=true);

# println("\n==> Backtracking LS.\n")

# @time x, v, primal, dual_gap, trajectoryBack = FrankWolfe.fw(f,grad,lmo,x0,max_iteration=k,
#     line_search=FrankWolfe.backtracking,L=100,print_iter=k/10,Emphasis=FrankWolfe.memory,verbose=true, trajectory=true);


println("\n==> Agnostic if function is too expensive for adaptive.\n")

x0 = deepcopy(x00)
@time x, v, primal, dual_gap, trajectoryBCG = FrankWolfe.bcg(
    f,
    grad,
    lmo,
    x0,
    max_iteration=k,
    line_search=FrankWolfe.adaptive,
    print_iter=k / 10,
    emphasis=FrankWolfe.memory,
    L=2,
    verbose=true,
    trajectory=true,
    Ktolerance=1.00,
    goodstep_tolerance=0.95,
    weight_purge_threshold=1e-10,
);

data = [trajectorySs, trajectoryAda, trajectoryBCG]
label = ["short step", "AFW", "BCG"]

FrankWolfe.plot_trajectories(data, label)

FrankWolfe.plot_trajectories(data[2:2], label[2:2])

using Plots
plot(getindex.(trajectoryAda, 4), xaxis=:log, yaxis=:log)


vs = getindex.(trajectoryBCG, 3)

plot(vs)
