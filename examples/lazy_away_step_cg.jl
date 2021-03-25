import FrankWolfe
import LinearAlgebra

# n = Int(1e1)
n = Int(1e4)
k = 5 * Int(1e3)
number_nonzero = 40

xpi = rand(n);
total = sum(xpi);
const xp = xpi # ./ total;

f(x) = LinearAlgebra.norm(x - xp)^2
function grad!(storage, x)
    @. storage = 2 * (x - xp)
end

lmo = FrankWolfe.KSparseLMO(number_nonzero, 1.0);
## alternative lmo
# lmo = FrankWolfe.ProbabilitySimplexOracle(1)
x0 = FrankWolfe.compute_extreme_point(lmo, ones(n));

@time x, v, primal, dual_gap, trajectorylazy, active_set = FrankWolfe.away_frank_wolfe(
    f,
    grad!,
    lmo,
    x0,
    max_iteration=k,
    line_search=FrankWolfe.adaptive,
    print_iter=k / 10,
    epsilon=1e-5,
    emphasis=FrankWolfe.memory,
    verbose=true,
    trajectory=true,
    lazy=true,
);

@time x, v, primal, dual_gap, trajectoryAFW, active_set = FrankWolfe.away_frank_wolfe(
    f,
    grad!,
    lmo,
    x0,
    max_iteration=k,
    line_search=FrankWolfe.adaptive,
    print_iter=k / 10,
    epsilon=1e-5,
    emphasis=FrankWolfe.memory,
    verbose=true,
    awaySteps=true,
    trajectory=true,
);

@time x, v, primal, dual_gap, trajectoryFW = FrankWolfe.away_frank_wolfe(
    f,
    grad!,
    lmo,
    x0,
    max_iteration=k,
    line_search=FrankWolfe.adaptive,
    print_iter=k / 10,
    emphasis=FrankWolfe.memory,
    verbose=true,
    epsilon=1e-5,
    trajectory=true,
    awaySteps=false,
);

data = [trajectorylazy, trajectoryAFW, trajectoryFW]
label = ["LAFW" "AFW" "FW"]

FrankWolfe.plot_trajectories(data, label,xscalelog=true)
