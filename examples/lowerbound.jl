import FrankWolfe
import LinearAlgebra


# n = Int(1e1)
n = Int(1e2)
k = Int(1e4)

xpi = zeros(n);
total = sum(xpi);
const xp = xpi # ./ total;

f(x) = LinearAlgebra.norm(x - xp)^2
function grad!(storage, x)
    @. storage = 2 * (x - xp)
end

lmo = FrankWolfe.ProbabilitySimplexOracle(1)
x0 = FrankWolfe.compute_extreme_point(lmo, zeros(n));

FrankWolfe.benchmark_oracles(f, grad!, ()-> rand(n), lmo; k=100)

@time x, v, primal, dual_gap, trajectory = FrankWolfe.fw(
    f,
    grad!,
    lmo,
    x0,
    max_iteration=k,
    line_search=FrankWolfe.agnostic,
    L=100,
    print_iter=k / 10,
    emphasis=FrankWolfe.memory,
    verbose=true,
    epsilon=1e-5,
    trajectory=true,
);


data = [trajectory]
label = ["FW"]

FrankWolfe.plot_trajectories(data, label, xscalelog=true)
