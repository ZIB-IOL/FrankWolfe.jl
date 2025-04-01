#=
This example demonstrates the use of the Blended Pairwise Conditional Gradient algorithm
with direct solve steps for a quadratic optimization problem over a sparse polytope which is not standard quadratic.

The example showcases how the algorithm balances between:
- Pairwise steps for efficient optimization
- Periodic direct solves for handling the quadratic objective
- Lazy (approximate) linear minimization steps for improved iteration complexity

It also demonstrates how to set up custom callbacks for tracking algorithm progress.
=#

using FrankWolfe
using LinearAlgebra
using Random
using ConjugateGradients
using IterativeSolvers

import HiGHS
import MathOptInterface as MOI

include("../examples/plot_utils.jl")

n = Int(1e2)
k = 10000

# s = rand(1:100)
s = 10
@info "Seed $s"
Random.seed!(s)

A = let
    A = randn(n, n)
    A' * A
end
@assert isposdef(A) == true

const y = Random.rand(Bool, n) * 0.6 .+ 0.3

# function f(x)
#     d = x - y
#     return dot(d, A, d)
# end

# function grad!(storage, x)
#     mul!(storage, A, x)
#     return mul!(storage, A, y, -2, 2)
# end


xpi = rand(n);
total = sum(xpi);

xp = xpi ./ total;

f(x) = norm(x - xp)^2
function grad!(storage, x)
    @. storage = 2 * (x - xp)
end

lmo = FrankWolfe.KSparseLMO(5, 500.0)

## other LMOs to try
#lmo = FrankWolfe.KSparseLMO(10, big"500.0")
# lmo = FrankWolfe.LpNormLMO{Float64,5}(100.0)
# lmo = FrankWolfe.ProbabilitySimplexOracle(100.0);
# lmo = FrankWolfe.UnitSimplexOracle(10000.0);

# vertices = [rand(n) for _ in 1:n^2]
# lmo = FrankWolfe.ConvexHullOracle(vertices)

x00 = FrankWolfe.compute_extreme_point(lmo, rand(n))


function build_callback(trajectory_arr)
    return function callback(state, active_set, args...)
        return push!(trajectory_arr, (FrankWolfe.callback_state(state)..., length(active_set)))
    end
end


function linear_operator_cg_solve(A_mat, b)
    A = (b, x) -> mul!(b, A_mat, x)
    x, _ = ConjugateGradients.cg(A, b; tol=1e-16, maxIter=10000)
    return x
end

traj_data = []
active_set = FrankWolfe.ActiveSetQuadraticLinearSolve(
    FrankWolfe.ActiveSet([(1.0, copy(x00))]),
    2I, -2xp,
    MOI.instantiate(MOI.OptimizerWithAttributes(HiGHS.Optimizer, MOI.Silent() => true)),
    scheduler=FrankWolfe.LogScheduler(start_time=10, scaling_factor=1),
    wolfe_step=true,
        cg_solve=linear_operator_cg_solve)

FrankWolfe.blended_pairwise_conditional_gradient(
    f,
    grad!,
    lmo,
    active_set,
    max_iteration=k,
    verbose=true,
    callback=build_callback(traj_data),
);

traj_data = []
active_set = FrankWolfe.ActiveSetQuadraticLinearSolve(
    FrankWolfe.ActiveSet([(1.0, copy(x00))]),
    2I, -2xp,
    MOI.instantiate(MOI.OptimizerWithAttributes(HiGHS.Optimizer, MOI.Silent() => true)),
    scheduler=FrankWolfe.LogScheduler(start_time=10, scaling_factor=1),
    wolfe_step=true,
    cg_solve=(A, b) -> IterativeSolvers.cg(A, b; abstol=1e-16, reltol=1e-16, maxiter=10000))

FrankWolfe.blended_pairwise_conditional_gradient(
    f,
    grad!,
    lmo,
    active_set,
    max_iteration=k,
    verbose=true,
    callback=build_callback(traj_data),
);