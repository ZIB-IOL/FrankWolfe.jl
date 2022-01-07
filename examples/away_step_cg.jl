include("activate.jl")

import LinearAlgebra


# n = Int(1e1)
n = Int(1e2)
k = Int(1e4)

xpi = rand(n);
total = sum(xpi);
const xp = xpi # ./ total;

f(x) = LinearAlgebra.norm(x - xp)^2
function grad!(storage, x)
    @. storage = 2 * (x - xp)
end

# problem with active set updates and the ksparselmo
lmo = FrankWolfe.KSparseLMO(40, 1.0);
# lmo = FrankWolfe.ProbabilitySimplexOracle(1)
x0 = FrankWolfe.compute_extreme_point(lmo, zeros(n));

FrankWolfe.benchmark_oracles(f, grad!, () -> rand(n), lmo; k=100)

x, v, primal, dual_gap, trajectory = FrankWolfe.frank_wolfe(
    f,
    grad!,
    lmo,
    x0,
    max_iteration=k,
    line_search=FrankWolfe.Adaptive(),
    L=100,
    print_iter=k / 10,
    emphasis=FrankWolfe.memory,
    verbose=true,
    epsilon=1e-5,
    trajectory=true,
);

x, v, primal, dual_gap, trajectory_away, active_set = FrankWolfe.away_frank_wolfe(
    f,
    grad!,
    lmo,
    x0,
    max_iteration=k,
    line_search=FrankWolfe.Adaptive(),
    L=100,
    print_iter=k / 10,
    epsilon=1e-5,
    emphasis=FrankWolfe.memory,
    verbose=true,
    away_steps=true,
    trajectory=true,
);

x, v, primal, dual_gap, trajectory_away_outplace, active_set = FrankWolfe.away_frank_wolfe(
    f,
    grad!,
    lmo,
    x0,
    max_iteration=k,
    line_search=FrankWolfe.Adaptive(),
    L=100,
    print_iter=k / 10,
    epsilon=1e-5,
    momentum=0.9,
    emphasis=FrankWolfe.blas,
    verbose=true,
    away_steps=true,
    trajectory=true,
);

data = [trajectory, trajectory_away, trajectory_away_outplace]
label = ["FW" "AFW" "MAFW"]

FrankWolfe.plot_trajectories(data, label)
