using Plots
using LinearAlgebra
using FrankWolfe
using JSON
using DelimitedFiles
using MAT

# Set the tolerance
eps = 1e-5

# NOTE: the data are random normal matrices with mean 0.05, not 0.1 as indicated in their paper
# we also generated additional datasets at larger scale and log-normal revenues
# 
# for large problem instances from https://zenodo.org/records/4836009
# see paper: https://arxiv.org/abs/2105.13913 

problem_instance = joinpath(@__DIR__, "data/syn_1000_800_10_50_1.mat")
W = MAT.matread(problem_instance)["W"]


# Set the maximum number of iterations
max_iteration = 5000

function build_objective(W)
    (n, p) = size(W)
    function f(x)
        -sum(log(dot(x, @view(W[:,t]))) for t in 1:p)
    end
    function ∇f(storage, x)
        storage .= 0
        for t in 1:p
            temp_rev = dot(x, @view(W[:,t]))
            @. storage -= @view(W[:,t]) ./ temp_rev
        end
        storage
    end
    (f, ∇f)
end

# lower bound on objective value
true_obj_value = -10.0

(f, ∇f) = build_objective(W)

lmo = FrankWolfe.ProbabilitySimplexOracle(1.0)
x0 = FrankWolfe.compute_extreme_point(lmo, rand(size(W, 1)))
storage = Vector{Float64}(undef, size(x0)...)

(x, v, primal_agnostic, dual_gap, traj_data_agnostic) = FrankWolfe.frank_wolfe(
    x -> f(x) - true_obj_value, ∇f, lmo, x0,
    verbose=true,
    trajectory=true,
    line_search=FrankWolfe.Agnostic(),
    max_iteration=max_iteration,
    gradient=storage,
    print_iter=max_iteration / 10,
    epsilon=eps,
)

(xback, v, primal_back, dual_gap, traj_data_backtracking) = FrankWolfe.frank_wolfe(
    x -> f(x) - true_obj_value, ∇f, lmo, x0,
    verbose=true,
    trajectory=true,
    line_search=FrankWolfe.Adaptive(),
    max_iteration=max_iteration,
    gradient=storage,
    print_iter=max_iteration / 10,
    epsilon=eps,
)

(xback, v, primal_back, dual_gap, traj_data_monotoninc) = FrankWolfe.frank_wolfe(
    x -> f(x) - true_obj_value, ∇f, lmo, x0,
    verbose=true,
    trajectory=true,
    line_search=FrankWolfe.MonotonicStepSize(),
    max_iteration=max_iteration,
    gradient=storage,
    print_iter=max_iteration / 10,
    epsilon=eps,
)

(xsecant, v, primal_secant, dual_gap, traj_data_secant) = FrankWolfe.frank_wolfe(
    x -> f(x) - true_obj_value, ∇f, lmo, x0,
    verbose=true,
    trajectory=true,
    line_search=FrankWolfe.Secant(tol=1e-12),
    max_iteration=max_iteration,
    gradient=storage,
    print_iter=max_iteration / 10,
    epsilon=eps,
)

# Plotting the trajectories
labels = ["Agnostic", "Adaptive", "Monotonic", "Secant"]
data = [traj_data_agnostic, traj_data_backtracking, traj_data_monotoninc, traj_data_secant]
plot_trajectories(data, labels,xscalelog=true)
