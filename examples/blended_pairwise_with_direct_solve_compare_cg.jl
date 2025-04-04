using FrankWolfe
using LinearAlgebra
using Random
using ConjugateGradients
using IterativeSolvers

import HiGHS
import MathOptInterface as MOI


n_runs = 5
n = Int(1e2)
k = 10000

s = 10
Random.seed!(s)


"""
    Define LMOs
"""

lmos = [
    FrankWolfe.KSparseLMO(5, 500.0),
    FrankWolfe.KSparseLMO(5, 1.0),
    FrankWolfe.KSparseLMO(10, 500.0),
    FrankWolfe.KSparseLMO(10, 1.0),
    FrankWolfe.KSparseLMO(50, 500.0),
    FrankWolfe.KSparseLMO(50, 1.0),
    FrankWolfe.KSparseLMO(100, 500.0),
    FrankWolfe.KSparseLMO(100, 1.0),
    FrankWolfe.LpNormLMO{Float64,5}(100.0),
    FrankWolfe.LpNormLMO{Float64,5}(1.0),
    FrankWolfe.LpNormLMO{Float64,2}(100.0),
    FrankWolfe.LpNormLMO{Float64,2}(1.0),
    FrankWolfe.UnitSimplexOracle(10000.0),
    FrankWolfe.UnitSimplexOracle(10.0),
]


"""
    Define solvers
"""

function cg_solve(x, A_mat, b)
    A = (b, x) -> mul!(b, A_mat, x)
    return ConjugateGradients.cg!(A, b, x; tol=1e-16, maxIter=1000)
end

function is_solve(x, A_mat, b)
    return IterativeSolvers.cg!(x, A_mat, b; abstol=1e-16, reltol=1e-16, maxiter=1000)
end

solvers =
    FrankWolfe.AffineMinSolver[FrankWolfe.UnsymmetricLPSolver(), FrankWolfe.SymmetricLPSolver()]

for warmstart in [0, 1, 2]
    push!(solvers, FrankWolfe.TranslationSolverCG(cg_solve, zeros(n), warmstart))
end

# for warmstart in [0,1,2]
#     push!(solvers, FrankWolfe.LagrangeSolverCG(
#         cg_solve,
#         zeros(n),
#         zeros(n),
#         warmstart
#     ))
# end

"""
    Run the experiment
"""

results = []
iterations = []
for r in 1:n_runs

    A = let
        A = randn(n, n)
        A' * A
    end

    # λ = eigvals(A)
    # @assert all(λ .< 50) "$(maximum(λ))"
    @assert isposdef(A)

    y = Random.rand(Bool, n) * 0.6 .+ 0.3

    Ay = A * y

    function f(x)
        d = x - y
        return dot(d, A, d)
    end

    function grad!(storage, x)
        mul!(storage, A, x)
        return mul!(storage, A, y, -2, 2)
    end

    result = Matrix{Float64}(undef, length(lmos), length(solvers))
    iteration_counts = Matrix{Int}(undef, length(lmos), length(solvers))
    for (i, lmo) in enumerate(lmos)
        x00 = FrankWolfe.compute_extreme_point(lmo, rand(n))

        for (j, solver) in enumerate(solvers)
            active_set = FrankWolfe.ActiveSetQuadraticLinearSolve(
                FrankWolfe.ActiveSet([(1.0, copy(x00))]),
                2 * A,
                -2 * Ay,
                MOI.instantiate(MOI.OptimizerWithAttributes(HiGHS.Optimizer, MOI.Silent() => true)),
                scheduler=FrankWolfe.LogScheduler(start_time=10, scaling_factor=1),
                wolfe_step=true,
                affine_min_solver=solver,
            )

            time_start = time()
            _, _, _, _, traj_data = FrankWolfe.blended_pairwise_conditional_gradient(
                f,
                grad!,
                lmo,
                active_set,
                max_iteration=k,
                trajectory=true,
            )
            result[i, j] = time() - time_start
            iteration_counts[i, j] = length(traj_data)
            print("\rProgress: $r/$n_runs, $i/$(length(lmos)), $j/$(length(solvers))")
        end
    end
    push!(results, result)
    push!(iterations, iteration_counts)
end

# Average results
average_results = sum(results) / n_runs
average_iterations = sum(iterations) / n_runs

# # Normalize results
# for i in 1:length(lmos)
#     max_result = maximum(average_results[i, :])
#     average_results[i, :] = average_results[i, :] ./ max_result
#     max_iterations = maximum(average_iterations[i, :])
#     average_iterations[i, :] = average_iterations[i, :] ./ max_iterations
# end

display(average_results)
display(average_iterations)

using Plots
gr()

xtick_labels = ["LP", "sym LP", "CG warmstart 0", "CG warmstart 1", "CG warmstart 2"]
ytick_labels = [
    "KSparse(5, 500)",
    "KSparse(5, 1)",
    "KSparse(10, 500)",
    "KSparse(10, 1)",
    "KSparse(50, 500)",
    "KSparse(50, 1)",
    "KSparse(100, 500)",
    "KSparse(100, 1)",
    "LpNorm(100, 5)",
    "LpNorm(1, 5)",
    "LpNorm(100, 2)",
    "LpNorm(1, 2)",
    "UnitSimplex(10000)",
    "UnitSimplex(10)",
]

h = heatmap(
    average_results,
    xlabel="Solvers",
    ylabel="LMOs",
    title="Blended Pairwise Conditional Gradient",
    xticks=(1:length(xtick_labels), xtick_labels),
    yticks=(1:length(ytick_labels), ytick_labels),
    size=(1000, 1000),
)

savefig(h, "blended_pairwise_with_direct_solve_compare_cg.png")
display(h)