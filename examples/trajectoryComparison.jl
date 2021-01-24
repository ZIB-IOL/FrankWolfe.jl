import FrankWolfe
import LinearAlgebra


n = Int(1e5)
k = 10000

xpi = rand(n);
total = sum(xpi);
const xp = xpi ./ total;

f(x) = LinearAlgebra.norm(x - xp)^2
grad(x) = 2 * (x - xp)
# better for memory consumption as we do coordinate-wise ops

function cf(x, xp)
    return @. LinearAlgebra.norm(x - xp)^2
end

function cgrad(x, xp)
    return @. 2 * (x - xp)
end

# lmo = FrankWolfe.KSparseLMO(100, 1.0)
lmo = FrankWolfe.LpNormLMO{Float64,1}(1.0)
# lmo = FrankWolfe.ProbabilitySimplexOracle(1.0);
# lmo = FrankWolfe.UnitSimplexOracle(1.0);
x00 = FrankWolfe.compute_extreme_point(lmo, zeros(n))
# print(x0)

FrankWolfe.benchmark_oracles(x -> cf(x, xp), x -> cgrad(x, xp), lmo, n; k=100, T=Float64)

# 1/t *can be* better than short step

println("\n==> Short Step rule - if you know L.\n")

x0 = copy(x00)
@time x, v, primal, dualGap, trajectorySs = FrankWolfe.fw(
    f,
    grad,
    lmo,
    x0,
    max_iteration=k,
    line_search=FrankWolfe.shortstep,
    L=2,
    print_iter=k / 10,
    Emphasis=FrankWolfe.memory,
    verbose=true,
    trajectory=true,
);

println("\n==> Short Step rule with momentum - if you know L.\n")

x0 = copy(x00)

@time x, v, primal, dualGap, trajectoryM = FrankWolfe.fw(
    f,
    grad,
    lmo,
    x0,
    max_iteration=k,
    line_search=FrankWolfe.shortstep,
    L=2,
    print_iter=k / 10,
    Emphasis=FrankWolfe.blas,
    verbose=true,
    trajectory=true,
    momentum=0.9,
);

println("\n==> Adaptive if you do not know L.\n")

x0 = copy(x00)

@time x, v, primal, dualGap, trajectoryAda = FrankWolfe.fw(
    f,
    grad,
    lmo,
    x0,
    max_iteration=k,
    line_search=FrankWolfe.adaptive,
    L=100,
    print_iter=k / 10,
    Emphasis=FrankWolfe.memory,
    verbose=true,
    trajectory=true,
);

# println("\n==> Goldenratio LS.\n")

# @time x, v, primal, dualGap, trajectoryGr = FrankWolfe.fw(f,grad,lmo,x0,max_iteration=k,
#     line_search=FrankWolfe.goldenratio,L=100,print_iter=k/10,Emphasis=FrankWolfe.memory,verbose=true, trajectory=true);

# println("\n==> Backtracking LS.\n")

# @time x, v, primal, dualGap, trajectoryBack = FrankWolfe.fw(f,grad,lmo,x0,max_iteration=k,
#     line_search=FrankWolfe.backtracking,L=100,print_iter=k/10,Emphasis=FrankWolfe.memory,verbose=true, trajectory=true);


println("\n==> Agnostic if function is too expensive for adaptive.\n")

x0 = copy(x00)

@time x, v, primal, dualGap, trajectoryAg = FrankWolfe.fw(
    f,
    grad,
    lmo,
    x0,
    max_iteration=k,
    line_search=FrankWolfe.agnostic,
    print_iter=k / 10,
    Emphasis=FrankWolfe.memory,
    verbose=true,
    trajectory=true,
);



data = [trajectorySs, trajectoryAda, trajectoryAg, trajectoryM]
label = ["short step" "adaptive" "agnostic" "momentum"]


FrankWolfe.plot_trajectories(data, label)
