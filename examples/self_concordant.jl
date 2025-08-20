using FrankWolfe
import LinearAlgebra

include("../examples/plot_utils.jl")

n = Int(1e5)
k = 1000

xpi = rand(n);
total = sum(xpi);
const xp = xpi ./ total;

f(x) = LinearAlgebra.norm(x - xp)^2

function grad!(storage, x)
    @. storage = 2 * (x - xp)
end


lmo = FrankWolfe.ProbabilitySimplexOracle(1);
x00 = FrankWolfe.compute_extreme_point(lmo, zeros(n));

FrankWolfe.benchmark_oracles(x -> f(x), (str, x) -> grad!(str, x), () -> randn(n), lmo; k=100)

println("\n==> Monotonic Step Size.\n")

x0 = deepcopy(x00)

@time x, v, primal, dual_gap, status, trajectory_monotonic = FrankWolfe.frank_wolfe(
    f,
    grad!,
    lmo,
    x0,
    max_iteration=k,
    line_search=FrankWolfe.MonotonicStepSize(),
    print_iter=k / 10,
    memory_mode=FrankWolfe.InplaceEmphasis(),
    verbose=true,
    trajectory=true,
);

println("\n==> Backtracking.\n")

x0 = deepcopy(x00)

@time x, v, primal, dual_gap, status, trajectory_backtracking = FrankWolfe.frank_wolfe(
    f,
    grad!,
    lmo,
    x0,
    max_iteration=k,
    line_search=FrankWolfe.Backtracking(),
    print_iter=k / 10,
    memory_mode=FrankWolfe.InplaceEmphasis(),
    verbose=true,
    trajectory=true,
);

println("\n==> Secant.\n")

x0 = deepcopy(x00)

@time x, v, primal, dual_gap, status, trajectory_secant = FrankWolfe.frank_wolfe(
    f,
    grad!,
    lmo,
    x0,
    max_iteration=k,
    line_search=FrankWolfe.Secant(),
    print_iter=k / 10,
    memory_mode=FrankWolfe.InplaceEmphasis(),
    verbose=true,
    trajectory=true,
);

data = [trajectory_monotonic, trajectory_backtracking, trajectory_secant]
label = ["monotonic", "backtracking", "secant"]

plot_trajectories(data, label, xscalelog=true)
