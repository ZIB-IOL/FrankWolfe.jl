using FrankWolfe
using LinearAlgebra

include("plot_utils.jl")

f(x) = dot(x[:, 1] - x[:, 2], x[:, 1] - x[:, 2])

function grad!(storage, x)
    g = 2 * hcat(x[:, 1] - x[:, 2], x[:, 2] - x[:, 1])
    @. storage = g
end

n = 100

lmo1 = FrankWolfe.ScaledBoundLInfNormBall(-ones(n), zeros(n))
lmo2 = FrankWolfe.ProbabilitySimplexOracle(1.0)
prod_lmo = FrankWolfe.ProductLMO((lmo1, lmo2))

x0 = compute_extreme_point(prod_lmo, ones(n, 2))

trajectories = []

# Example for creating a custome block coordinate update order
struct CustomOrder <: FrankWolfe.BlockCoordinateUpdateOrder end

function FrankWolfe.select_update_indices(::CustomOrder, l)
    return [rand() < 1 / n ? 1 : 2 for _ in 1:l]
end

for order in [
    FrankWolfe.FullUpdate(),
    FrankWolfe.CyclicUpdate(),
    FrankWolfe.StochasticUpdate(),
    CustomOrder(),
]

    _, _, _, _, traj_data = FrankWolfe.block_coordinate_frank_wolfe(
        f,
        grad!,
        prod_lmo,
        x0;
        verbose=true,
        trajectory=true,
        update_order=order,
    )
    push!(trajectories, traj_data)
end

labels = ["Full update", "Cyclic order", "Stochstic order", "Custom order"]
plot_trajectories(trajectories, labels, xscalelog=true)
