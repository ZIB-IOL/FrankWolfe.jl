#= 

Example demonstrating sparsity control by means of the "K"-factor passed to the lazy AFW variant

A larger K >= 1 favors sparsity by favoring optimization over the current active set rather than
adding a new FW vertex.

The default for AFW is K = 2.0

=#

include("activate.jl")

using LinearAlgebra
using Random

n = Int(1e3)
k = 10000

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

const lmo = FrankWolfe.KSparseLMO(5, 1.0)

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
    line_search=FrankWolfe.Shortstep(),
    L=2,
    print_iter=k / 10,
    emphasis=FrankWolfe.memory,
    verbose=true,
    trajectory=true,
);


trajectory_afw = []
callback = build_callback(trajectory_afw)

x0 = deepcopy(x00)
@time x, v, primal, dual_gap, _ = FrankWolfe.away_frank_wolfe(
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
    callback=callback,
);


trajectory_lafw = []
callback = build_callback(trajectory_lafw)

x0 = deepcopy(x00)
@time x, v, primal, dual_gap, _ = FrankWolfe.away_frank_wolfe(
    f,
    grad!,
    lmo,
    x0,
    max_iteration=k,
    line_search=FrankWolfe.Adaptive(),
    print_iter=k / 10,
    emphasis=FrankWolfe.memory,
    verbose=true,
    lazy=true,
    trajectory=true,
    callback=callback,
);

trajectoryBPCG = []
callback = build_callback(trajectoryBPCG)

x0 = deepcopy(x00)
@time x, v, primal, dual_gap, _ = FrankWolfe.blended_pairwise_conditional_gradient(
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
    callback=callback,
);


trajectoryLBPCG = []
callback = build_callback(trajectoryLBPCG)

x0 = deepcopy(x00)
@time x, v, primal, dual_gap, _ = FrankWolfe.blended_pairwise_conditional_gradient(
    f,
    grad!,
    lmo,
    x0,
    max_iteration=k,
    line_search=FrankWolfe.Adaptive(),
    print_iter=k / 10,
    emphasis=FrankWolfe.memory,
    verbose=true,
    lazy=true,
    trajectory=true,
    callback=callback,
);


# Reduction primal/dual error vs. sparsity of solution

dataSparsity =
    [trajectory_afw, trajectory_lafw, trajectoryBPCG, trajectoryLBPCG]
labelSparsity = ["AFW", "LAFW", "BPCG", "LBPCG"]

FrankWolfe.plot_sparsity(dataSparsity, labelSparsity, legend_position=:topright)
