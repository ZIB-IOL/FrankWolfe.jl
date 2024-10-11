#=
This example demonstrates the use of the Blended Pairwise Conditional Gradient algorithm
with direct solve steps for a quadratic optimization problem over a sparse polytope. 

Note the special structure of f(x) =  norm(x - x0)^2 that we assume here

The example showcases how the algorithm balances between:
- Pairwise steps for efficient optimization
- Periodic direct solves for handling the quadratic objective
- Lazy (approximate) linear minimization steps for improved iteration complexity

It also demonstrates how to set up custom callbacks for tracking algorithm progress.
=#

using FrankWolfe
using LinearAlgebra
using Random

import HiGHS
import MathOptInterface as MOI

# lp_solver = GLPK.Optimizer
lp_solver = HiGHS.Optimizer
# lp_solver = Clp.Optimizer # buggy / does not work properly


include("../examples/plot_utils.jl")

n = Int(1e4)
k = 5000

# s = rand(1:100)
s = 10
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

lmo = FrankWolfe.KSparseLMO(5, 1.0)

## other LMOs to try
# lmo_big = FrankWolfe.KSparseLMO(100, big"1.0")
# lmo = FrankWolfe.LpNormLMO{Float64,5}(1.0)
# lmo = FrankWolfe.ProbabilitySimplexOracle(1.0);
# lmo = FrankWolfe.UnitSimplexOracle(1.0);

const x00 = FrankWolfe.compute_extreme_point(lmo, rand(n))

function build_callback(trajectory_arr)
    return function callback(state, active_set, args...)
        return push!(trajectory_arr, (FrankWolfe.callback_state(state)..., length(active_set)))
    end
end


FrankWolfe.benchmark_oracles(f, grad!, () -> randn(n), lmo; k=100)

trajectoryBPCG_standard = []
callback = build_callback(trajectoryBPCG_standard)

@time x, v, primal, dual_gap, _ = FrankWolfe.blended_pairwise_conditional_gradient(
    f,
    grad!,
    lmo,
    copy(x00),
    max_iteration=k,
    line_search=FrankWolfe.Shortstep(2.0),
    print_iter=k / 10,
    memory_mode=FrankWolfe.InplaceEmphasis(),
    verbose=true,
    trajectory=true,
    callback=callback,
    squadratic=false,
);

trajectoryBPCG_quadratic = []
callback = build_callback(trajectoryBPCG_quadratic)

@time x, v, primal, dual_gap, _ = FrankWolfe.blended_pairwise_conditional_gradient(
    f,
    grad!,
    lmo,
    copy(x00),
    max_iteration=k,
    line_search=FrankWolfe.Shortstep(2.0),
    print_iter=k / 10,
    memory_mode=FrankWolfe.InplaceEmphasis(),
    verbose=true,
    trajectory=true,
    callback=callback,
    squadratic=true,
    lp_solver=lp_solver,
);

# Just using that qudratic (more expensive)
trajectoryBPCG_quadratic_nosquad = []
callback = build_callback(trajectoryBPCG_quadratic_nosquad)

@time x, v, primal, dual_gap, _ = FrankWolfe.blended_pairwise_conditional_gradient(
    f,
    grad!,
    lmo,
    copy(x00),
    max_iteration=k,
    line_search=FrankWolfe.Shortstep(2.0),
    print_iter=k / 10,
    memory_mode=FrankWolfe.InplaceEmphasis(),
    verbose=true,
    trajectory=true,
    callback=callback,
    quadratic=true,
    squadratic=false,
    lp_solver=lp_solver,
);

# Just projection quadratic
trajectoryBPCG_quadratic_as = []
callback = build_callback(trajectoryBPCG_quadratic_as)

as_quad = FrankWolfe.ActiveSetQuadratic([(1.0, copy(x00))], 2 * LinearAlgebra.I, -2xp)
@time x, v, primal, dual_gap, _ = FrankWolfe.blended_pairwise_conditional_gradient(
    f,
    grad!,
    lmo,
    as_quad,
    max_iteration=k,
    line_search=FrankWolfe.Shortstep(2.0),
    print_iter=k / 10,
    memory_mode=FrankWolfe.InplaceEmphasis(),
    verbose=true,
    trajectory=true,
    callback=callback,
);

as_quad = FrankWolfe.ActiveSetQuadratic(
    [(1.0, copy(x00))],
    2 * LinearAlgebra.I, -2xp,
)

# with quadratic active set
trajectoryBPCG_quadratic_as = []
callback = build_callback(trajectoryBPCG_quadratic_as)
@time x, v, primal, dual_gap, _ = FrankWolfe.blended_pairwise_conditional_gradient(
    f,
    grad!,
    lmo,
    as_quad,
    max_iteration=k,
    line_search=FrankWolfe.Shortstep(2.0),
    print_iter=k / 10,
    memory_mode=FrankWolfe.InplaceEmphasis(),
    verbose=true,
    trajectory=true,
    callback=callback,
);

as_quad_direct = FrankWolfe.ActiveSetQuadraticLinearSolve(
    [(1.0, copy(x00))],
    2 * LinearAlgebra.I, -2xp,
    MOI.instantiate(MOI.OptimizerWithAttributes(HiGHS.Optimizer, MOI.Silent() => true)),
)

# with LP acceleration
trajectoryBPCG_quadratic_direct = []
callback = build_callback(trajectoryBPCG_quadratic_direct)

@time x, v, primal, dual_gap, _ = FrankWolfe.blended_pairwise_conditional_gradient(
    f,
    grad!,
    lmo,
    as_quad_direct,
    max_iteration=k,
    line_search=FrankWolfe.Shortstep(2.0),
    print_iter=k / 10,
    memory_mode=FrankWolfe.InplaceEmphasis(),
    verbose=true,
    trajectory=true,
    callback=callback,
);

as_quad_direct_generic = FrankWolfe.ActiveSetQuadraticLinearSolve(
    [(1.0, copy(x00))],
    2 * Diagonal(ones(length(xp))), -2xp,
    MOI.instantiate(MOI.OptimizerWithAttributes(HiGHS.Optimizer, MOI.Silent() => true)),
)

# with LP acceleration
trajectoryBPCG_quadratic_direct_generic = []
callback = build_callback(trajectoryBPCG_quadratic_direct_generic)

@time x, v, primal, dual_gap, _ = FrankWolfe.blended_pairwise_conditional_gradient(
    f,
    grad!,
    lmo,
    as_quad_direct_generic,
    max_iteration=k,
    line_search=FrankWolfe.Shortstep(2.0),
    print_iter=k / 10,
    memory_mode=FrankWolfe.InplaceEmphasis(),
    verbose=true,
    trajectory=true,
    callback=callback,
);


# Update the data and labels for plotting
dataSparsity = [trajectoryBPCG_standard, trajectoryBPCG_quadratic, trajectoryBPCG_quadratic_as, trajectoryBPCG_quadratic_direct, trajectoryBPCG_quadratic_direct_generic]
labelSparsity = ["BPCG (Standard)", "BPCG (Specific Direct)", "AS_Quad", "Reloaded", "Reloaded_generic"]

# Plot sparsity
# plot_sparsity(dataSparsity, labelSparsity, legend_position=:topright)

# Plot trajectories
plot_trajectories(dataSparsity, labelSparsity, xscalelog=false)
