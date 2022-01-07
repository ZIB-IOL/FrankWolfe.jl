include("activate.jl")

import LinearAlgebra


n = Int(1e5)
k = 1000

xpi = rand(n);
total = sum(xpi);
const xp = xpi ./ total;

f(x) = LinearAlgebra.norm(x - xp)^2

function grad!(storage, x)
    @. storage = 2 * (x - xp)
end

lmo = FrankWolfe.KSparseLMO(40, 1.0);
x00 = FrankWolfe.compute_extreme_point(lmo, zeros(n));

FrankWolfe.benchmark_oracles(x -> f(x), (str, x) -> grad!(str, x), () -> randn(n), lmo; k=100)

println("\n==> Short Step rule - if you know L.\n")

x0 = deepcopy(x00)

@time x, v, primal, dual_gap, trajectory_shortstep = FrankWolfe.frank_wolfe(
    f,
    grad!,
    lmo,
    x0,
    max_iteration=k,
    line_search=FrankWolfe.Shortstep(),
    L=2,
    print_iter=k / 10,
    emphasis=FrankWolfe.memory,
    verbose=true,
    trajectory=true,
);

println("\n==> Adaptive if you do not know L.\n")

x0 = deepcopy(x00)

@time x, v, primal, dual_gap, trajectory_adaptive = FrankWolfe.frank_wolfe(
    f,
    grad!,
    lmo,
    x0,
    max_iteration=k,
    line_search=FrankWolfe.Adaptive(),
    print_iter=k / 10,
    emphasis=FrankWolfe.memory,
    verbose=true,
    trajectory=true,
);

println("\n==> Agnostic if function is too expensive for adaptive.\n")

x0 = deepcopy(x00)

@time x, v, primal, dual_gap, trajectory_agnostic = FrankWolfe.frank_wolfe(
    f,
    grad!,
    lmo,
    x0,
    max_iteration=k,
    line_search=FrankWolfe.Agnostic(),
    print_iter=k / 10,
    emphasis=FrankWolfe.memory,
    verbose=true,
    trajectory=true,
);

data = [trajectory_shortstep, trajectory_adaptive, trajectory_agnostic]
label = ["short step", "adaptive", "agnostic"]


FrankWolfe.plot_trajectories(data, label, xscalelog=true)
