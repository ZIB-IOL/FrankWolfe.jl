import FrankWolfe
using LinearAlgebra
using Random


include("../examples/plot_utils.jl")

Random.seed!(100)

n = 500
lmo = FrankWolfe.ConvexHullOracle([rand(Float64, (n,)) .+ 1.0 for _ in 1:n])
lmo2 = FrankWolfe.ConvexHullOracle([rand(Float64, (n,)) .- 1.0 for _ in 1:n])


trajectories = []

methods = [
    FrankWolfe.frank_wolfe,
    FrankWolfe.blended_pairwise_conditional_gradient,
    FrankWolfe.blended_pairwise_conditional_gradient,
]

for (i, m) in enumerate(methods)
    if i == 1
        x, _, _, _, traj_data = FrankWolfe.alternating_projections(
            (lmo, lmo2),
            ones(n);
            verbose=true,
            print_iter=100,
            trajectory=true,
            proj_method=m,
        )
    elseif i == 2
        x, _, _, _, traj_data = FrankWolfe.alternating_projections(
            (lmo, lmo2),
            ones(n);
            verbose=true,
            print_iter=100,
            trajectory=true,
            proj_method=m,
            reuse_active_set=false,
            lazy=true,
        )
    else
        x, _, _, _, traj_data = FrankWolfe.alternating_projections(
            (lmo, lmo2),
            ones(n);
            verbose=true,
            print_iter=100,
            trajectory=true,
            proj_method=m,
            reuse_active_set=true,
            lazy=true,
        )
    end
    push!(trajectories, traj_data)
end



labels = ["FW", "BPCG", "BPCG (reuse)"]

plot_trajectories(trajectories, labels, xscalelog=true)
