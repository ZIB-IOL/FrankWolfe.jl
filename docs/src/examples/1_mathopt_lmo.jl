# # Comparison with MathOptInterface on a Probability Simplex

# In this example, we project a random point onto a probability simplex with the Frank-Wolfe algorithm using
# either the specialized LMO defined in the package or a generic LP formulation using `MathOptInterface.jl` (MOI) and
# `GLPK` as underlying LP solver.
# It can be found as Example 4.4 [in the paper](https://arxiv.org/abs/2104.06675).

using FrankWolfe

using LinearAlgebra
using LaTeXStrings

using Plots

using JuMP
const MOI = JuMP.MOI

import GLPK

n = Int(1e3)
k = 10000

xpi = rand(n);
total = sum(xpi);
const xp = xpi ./ total;

f(x) = norm(x - xp)^2
function grad!(storage, x)
    @. storage = 2 * (x - xp)
    return nothing
end

lmo_radius = 2.5
lmo = FrankWolfe.FrankWolfe.ProbabilitySimplexOracle(lmo_radius)

x00 = FrankWolfe.compute_extreme_point(lmo, zeros(n))
gradient = collect(x00)

x_lmo, v, primal, dual_gap, trajectory_lmo = FrankWolfe.frank_wolfe(
    f,
    grad!,
    lmo,
    collect(copy(x00)),
    max_iteration=k,
    line_search=FrankWolfe.Shortstep(2.0),
    print_iter=k / 10,
    memory_mode=FrankWolfe.InplaceEmphasis(),
    verbose=false,
    trajectory=true,
);

# Create a MathOptInterface Optimizer and build the same linear constraints:

o = GLPK.Optimizer()
x = MOI.add_variables(o, n)

for xi in x
    MOI.add_constraint(o, xi, MOI.GreaterThan(0.0))
end

MOI.add_constraint(
    o,
    MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(1.0, x), 0.0),
    MOI.EqualTo(lmo_radius),
)

lmo_moi = FrankWolfe.MathOptLMO(o)

x, v, primal, dual_gap, trajectory_moi = FrankWolfe.frank_wolfe(
    f,
    grad!,
    lmo_moi,
    collect(copy(x00)),
    max_iteration=k,
    line_search=FrankWolfe.Shortstep(2.0),
    print_iter=k / 10,
    memory_mode=FrankWolfe.InplaceEmphasis(),
    verbose=false,
    trajectory=true,
);

# Alternatively, we can use one of the modelling interfaces based on `MOI` to formulate the LP.
# The following example builds the same set of constraints using `JuMP`:

m = JuMP.Model(GLPK.Optimizer)
@variable(m, y[1:n] ≥ 0)

@constraint(m, sum(y) == lmo_radius)

lmo_jump = FrankWolfe.MathOptLMO(m.moi_backend)

x, v, primal, dual_gap, trajectory_jump = FrankWolfe.frank_wolfe(
    f,
    grad!,
    lmo_jump,
    collect(copy(x00)),
    max_iteration=k,
    line_search=FrankWolfe.Shortstep(2.0),
    print_iter=k / 10,
    memory_mode=FrankWolfe.InplaceEmphasis(),
    verbose=false,
    trajectory=true,
);

x_lmo, v, primal, dual_gap, trajectory_lmo_blas = FrankWolfe.frank_wolfe(
    f,
    grad!,
    lmo,
    x00,
    max_iteration=k,
    line_search=FrankWolfe.Shortstep(2.0),
    print_iter=k / 10,
    memory_mode=FrankWolfe.OutplaceEmphasis(),
    verbose=false,
    trajectory=true,
);

x, v, primal, dual_gap, trajectory_jump_blas = FrankWolfe.frank_wolfe(
    f,
    grad!,
    lmo_jump,
    x00,
    max_iteration=k,
    line_search=FrankWolfe.Shortstep(2.0),
    print_iter=k / 10,
    memory_mode=FrankWolfe.OutplaceEmphasis(),
    verbose=false,
    trajectory=true,
);


iteration_list = [[x[1] + 1 for x in trajectory_lmo], [x[1] + 1 for x in trajectory_moi]]
time_list = [[x[5] for x in trajectory_lmo], [x[5] for x in trajectory_moi]]
primal_gap_list = [[x[2] for x in trajectory_lmo], [x[2] for x in trajectory_moi]]
dual_gap_list = [[x[4] for x in trajectory_lmo], [x[4] for x in trajectory_moi]]

label = [L"\textrm{Closed-form LMO}", L"\textrm{MOI LMO}"]

plot_results(
    [primal_gap_list, primal_gap_list, dual_gap_list, dual_gap_list],
    [iteration_list, time_list, iteration_list, time_list],
    label,
    ["", "", L"\textrm{Iteration}", L"\textrm{Time}"],
    [L"\textrm{Primal Gap}", "", L"\textrm{Dual Gap}", ""],
    xscalelog=[:log, :identity, :log, :identity],
    yscalelog=[:log, :log, :log, :log],
    legend_position=[:bottomleft, nothing, nothing, nothing]
)
