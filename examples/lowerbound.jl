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

NOTE:
1. ignore the timing graphs 
2. as the primal gap lower bounds the dual gap we also plot the primal gap lower bound in the dual gap graph
3. AFW violates the bound in the very first round which is due to the initialization in AFW starting with two vertices
4. all methods that call an LMO at most once in an iteration are subject to this lower bound
5. the objective is strongly convex, this implies limitations also for strongly convex functions (see for a discussion https://arxiv.org/abs/1906.07867)

=#

import FrankWolfe
import LinearAlgebra


# n = Int(1e1)
n = Int(1e2)
k = Int(1e3)

xp = 1/n * ones(n);

f(x) = LinearAlgebra.norm(x - xp)^2
function grad!(storage, x)
    @. storage = 2 * (x - xp)
end

lmo = FrankWolfe.ProbabilitySimplexOracle(1)
x0 = FrankWolfe.compute_extreme_point(lmo, zeros(n));

FrankWolfe.benchmark_oracles(f, grad!, ()-> rand(n), lmo; k=100)

@time x, v, primal, dual_gap, trajectory = FrankWolfe.frank_wolfe(
    f,
    grad!,
    lmo,
    x0,
    max_iteration=k,
    line_search=FrankWolfe.agnostic,
    L=2,
    print_iter=k / 10,
    emphasis=FrankWolfe.memory,
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
    line_search=FrankWolfe.adaptive,
    L=2,
    print_iter=k / 10,
    emphasis=FrankWolfe.memory,
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
    line_search=FrankWolfe.adaptive,
    L=2,
    print_iter=k / 10,
    emphasis=FrankWolfe.memory,
    verbose=true,
    epsilon=1e-5,
    trajectory=true,
);

# define lower bound 

trajLowerbound = []
for i in 1:n-1
    push!(
        trajLowerbound,
        (i, 1/i - 1/n, NaN, 1/i - 1/n, NaN),
    )
end


data = [trajectory, trajectoryAFW, trajectoryBCG,trajLowerbound]
label = ["FW", "AFW", "BCG", "Lowerbound"]

FrankWolfe.plot_trajectories(data, label, xscalelog=true,legendPosition=:bottomleft)
