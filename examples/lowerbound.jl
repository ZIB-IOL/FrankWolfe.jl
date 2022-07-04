#=

Lower bound instance from

http://proceedings.mlr.press/v28/jaggi13.pdf

and

https://arxiv.org/abs/1309.5550

Example instance is to minimize || x ||^2 over the probability simplex conv(e_1, ..., e_n)

Then the primal gap is lower bounded by 1/k - 1/n in iteration k as the optimal solution has value 1/n attained by
the (1/n, ..., 1/n) vector and in iteration k we have picked up at most k vertices from the simplex lower bounding the
primal value by 1/k.

Here: slightly rewritten to consider || x - (1/n, ..., 1/n) ||^2 so that the function value becomes directly the
primal gap (squared)

Three runs are compared:
1. Frank-Wolfe with traditional step-size rule
2. Away-step Frank-Wolfe with adaptive step-size rule
3. Blended Conditional Gradients with adaptive step-size rule

NOTE:
1. ignore the timing graphs
2. as the primal gap lower bounds the dual gap we also plot the primal gap lower bound in the dual gap graph
3. AFW violates the bound in the very first round which is due to the initialization in AFW starting with two vertices
4. all methods that call an LMO at most once in an iteration are subject to this lower bound
5. the objective is strongly convex, this implies limitations also for strongly convex functions (see for a discussion https://arxiv.org/abs/1906.07867)

=#

using FrankWolfe
using ProgressMeter
using Arpack
using Plots
using DoubleFloats
using ReverseDiff

import LinearAlgebra


# n = Int(1e1)
n = Int(1e2)
k = Int(1e3)

xp = 1 / n * ones(n);

# definition of objective
f(x) = LinearAlgebra.norm(x - xp)^2

# definition of gradient
function grad!(storage, x)
    @. storage = 2 * (x - xp)
end

# define LMO and do initial call to obtain starting point
lmo = FrankWolfe.ProbabilitySimplexOracle(1)
x0 = FrankWolfe.compute_extreme_point(lmo, zeros(n));

# simple benchmarking of oracles to get an idea how expensive each component is
FrankWolfe.benchmark_oracles(f, grad!, () -> rand(n), lmo; k=100)

@time x, v, primal, dual_gap, trajectory = FrankWolfe.frank_wolfe(
    f,
    grad!,
    lmo,
    x0,
    max_iteration=k,
    line_search=FrankWolfe.Agnostic(),
    print_iter=k / 10,
    memory_mode=FrankWolfe.InplaceEmphasis(),
    verbose=true,
    epsilon=1e-5,
    trajectory=true,
);

@time x, v, primal, dual_gap, trajectoryAFW = FrankWolfe.away_frank_wolfe(
    f,
    grad!,
    lmo,
    x0,
    max_iteration=k,
    line_search=FrankWolfe.Adaptive(),
    print_iter=k / 10,
    memory_mode=FrankWolfe.InplaceEmphasis(),
    verbose=true,
    epsilon=1e-5,
    trajectory=true,
);

@time x, v, primal, dual_gap, trajectoryBCG = FrankWolfe.blended_conditional_gradient(
    f,
    grad!,
    lmo,
    x0,
    max_iteration=k,
    line_search=FrankWolfe.Adaptive(),
    print_iter=k / 10,
    memory_mode=FrankWolfe.InplaceEmphasis(),
    verbose=true,
    epsilon=1e-5,
    trajectory=true,
);

# define lower bound

trajLowerbound = []
for i in 1:n-1
    push!(trajLowerbound, (i, 1 / i - 1 / n, NaN, 1 / i - 1 / n, NaN))
end


data = [trajectory, trajectoryAFW, trajectoryBCG, trajLowerbound]
label = ["FW", "AFW", "BCG", "Lowerbound"]

# ignore the timing plots - they are not relevant for this example
plot_trajectories(data, label, xscalelog=true, legend_position=:bottomleft)
