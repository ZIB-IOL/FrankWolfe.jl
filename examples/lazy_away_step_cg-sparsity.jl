#=

Example demonstrating sparsity control by means of the `lazy_tolerance`-factor passed to the lazy AFW variant

A larger lazy_tolerance >= 1 favors sparsity by favoring optimization over the current active set rather than
adding a new FW vertex.

The default for AFW is lazy_tolerance = 2.0

=#

using FrankWolfe
using ProgressMeter
using Arpack
using Plots
using DoubleFloats
using ReverseDiff

using LinearAlgebra
using Random

n = Int(1e4)
k = 1000

s = rand(1:100)
@info "Seed $s"
Random.seed!(s)

xpi = rand(n);
total = sum(xpi);

# here the optimal solution lies in the interior if you want an optimal solution on a face and not the interior use:
# const xp = xpi;

const xp = xpi ./ total;

f(x) = norm(x - xp)^2
function grad!(storage, x)
    @. storage = 2 * (x - xp)
end

const lmo = FrankWolfe.KSparseLMO(100, 1.0)

## other LMOs to try
# lmo_big = FrankWolfe.KSparseLMO(100, big"1.0")
# lmo = FrankWolfe.LpNormLMO{Float64,5}(1.0)
# lmo = FrankWolfe.ProbabilitySimplexOracle(1.0);
# lmo = FrankWolfe.UnitSimplexOracle(1.0);

const x00 = FrankWolfe.compute_extreme_point(lmo, rand(n))


## example with BirkhoffPolytopeLMO - uses square matrix.
# const lmo = FrankWolfe.BirkhoffPolytopeLMO()
# cost = rand(n, n)
# const x00 = FrankWolfe.compute_extreme_point(lmo, cost)


function build_callback(trajectory_arr)
    return function callback(state)
        return push!(trajectory_arr, (Tuple(state)[1:5]..., length(state.active_set)))
    end
end



FrankWolfe.benchmark_oracles(f, grad!, () -> randn(n), lmo; k=100)

x0 = deepcopy(x00)
@time x, v, primal, dual_gap, trajectory_shortstep = FrankWolfe.frank_wolfe(
    f,
    grad!,
    lmo,
    x0,
    max_iteration=k,
    line_search=FrankWolfe.Shortstep(2.0),
    print_iter=k / 10,
    memory_mode=FrankWolfe.InplaceEmphasis(),
    verbose=true,
    trajectory=true,
);


trajectory_adaptive = []
callback = build_callback(trajectory_adaptive)

x0 = deepcopy(x00)
@time x, v, primal, dual_gap, _ = FrankWolfe.away_frank_wolfe(
    f,
    grad!,
    lmo,
    x0,
    max_iteration=k,
    line_search=FrankWolfe.Adaptive(),
    print_iter=k / 10,
    memory_mode=FrankWolfe.InplaceEmphasis(),
    verbose=true,
    trajectory=true,
    callback=callback,
);


println("\n==> Lazy AFW.\n")

trajectory_adaptiveLoc15 = []
callback = build_callback(trajectory_adaptiveLoc15)

x0 = deepcopy(x00)
@time x, v, primal, dual_gap, _ = FrankWolfe.away_frank_wolfe(
    f,
    grad!,
    lmo,
    x0,
    max_iteration=k,
    line_search=FrankWolfe.Adaptive(),
    print_iter=k / 10,
    memory_mode=FrankWolfe.InplaceEmphasis(),
    verbose=true,
    lazy=true,
    lazy_tolerance=1.5,
    trajectory=true,
    callback=callback,
);


trajectory_adaptiveLoc2 = []
callback = build_callback(trajectory_adaptiveLoc2)

x0 = deepcopy(x00)
@time x, v, primal, dual_gap, _ = FrankWolfe.away_frank_wolfe(
    f,
    grad!,
    lmo,
    x0,
    max_iteration=k,
    line_search=FrankWolfe.Adaptive(),
    print_iter=k / 10,
    memory_mode=FrankWolfe.InplaceEmphasis(),
    verbose=true,
    lazy=true,
    lazy_tolerance=2.0,
    trajectory=true,
    callback=callback,
);


trajectory_adaptiveLoc4 = []
callback = build_callback(trajectory_adaptiveLoc4)

x0 = deepcopy(x00)
@time x, v, primal, dual_gap, _ = FrankWolfe.away_frank_wolfe(
    f,
    grad!,
    lmo,
    x0,
    max_iteration=k,
    line_search=FrankWolfe.Adaptive(),
    print_iter=k / 10,
    memory_mode=FrankWolfe.InplaceEmphasis(),
    verbose=true,
    lazy_tolerance=4.0,
    lazy=true,
    trajectory=true,
    callback=callback,
);

trajectory_adaptiveLoc10 = []
callback = build_callback(trajectory_adaptiveLoc10)

x0 = deepcopy(x00)
@time x, v, primal, dual_gap, _ = FrankWolfe.away_frank_wolfe(
    f,
    grad!,
    lmo,
    x0,
    max_iteration=k,
    line_search=FrankWolfe.Adaptive(),
    print_iter=k / 10,
    memory_mode=FrankWolfe.InplaceEmphasis(),
    lazy=true,
    lazy_tolerance=10.0,
    verbose=true,
    trajectory=true,
    callback=callback,
);


# Reduction primal/dual error vs. sparsity of solution

dataSparsity =
    [trajectory_adaptive, trajectory_adaptiveLoc15, trajectory_adaptiveLoc2, trajectory_adaptiveLoc4, trajectory_adaptiveLoc10]
labelSparsity = ["AFW", "LAFW-K-1.5", "LAFW-K-2.0", "LAFW-K-4.0", "LAFW-K-10.0"]

plot_sparsity(dataSparsity, labelSparsity, legend_position=:topright)
