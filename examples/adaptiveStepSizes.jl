import FrankWolfe
import LinearAlgebra


n = Int(1e5)
k = 1000
rescale = 400

xpi = rand(n);
total = sum(xpi);
const xp = xpi ./ total;

f(x) = rescale * LinearAlgebra.norm(x - xp)^2
grad(x) = rescale * 2 * (x - xp)

lmo = FrankWolfe.KSparseLMO(40, 1);
x00 = FrankWolfe.compute_extreme_point(lmo, zeros(n));

FrankWolfe.benchmark_oracles(f, grad, lmo, n; k=100, T=Float64)

println("\n==> Short Step rule - if you know L.\n")

x0 = deepcopy(x00)

@time x, v, primal, dual_gap, trajectorySs = FrankWolfe.fw(
    f,
    grad,
    lmo,
    x0,
    max_iteration=k,
    line_search=FrankWolfe.shortstep,
    L=2*rescale,
    print_iter=k / 10,
    emphasis=FrankWolfe.memory,
    verbose=true,
    trajectory=true
);

println("\n==> Adaptive if you do not know L.\n")

x0 = deepcopy(x00)

@time x, v, primal, dual_gap, trajectoryAda = FrankWolfe.fw(
    f,
    grad,
    lmo,
    x0,
    max_iteration=k,
    line_search=FrankWolfe.adaptive,
    L=2,
    print_iter=k / 10,
    emphasis=FrankWolfe.memory,
    verbose=true,
    trajectory=true
);

@time x, v, primal, dual_gap, trajectoryAdaL = FrankWolfe.fw(
    f,
    grad,
    lmo,
    x0,
    max_iteration=k,
    line_search=FrankWolfe.adaptive,
    L=10*rescale,
    print_iter=k / 10,
    emphasis=FrankWolfe.memory,
    verbose=true,
    trajectory=true
);

println("\n==> Agnostic if function is too expensive for adaptive.\n")

x0 = deepcopy(x00)

@time x, v, primal, dual_gap, trajectoryAg = FrankWolfe.fw(
    f,
    grad,
    lmo,
    x0,
    max_iteration=k,
    line_search=FrankWolfe.agnostic,
    print_iter=k / 10,
    emphasis=FrankWolfe.memory,
    verbose=true,
    trajectory=true
);

data = [trajectorySs, trajectoryAda, trajectoryAdaL, trajectoryAg]
label = ["short step" "adaptive" "adaptiveL" "agnostic"]


FrankWolfe.plot_trajectories(data, label)
