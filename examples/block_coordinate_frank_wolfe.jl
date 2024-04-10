using FrankWolfe
using LinearAlgebra

include("plot_utils.jl")

f(x) = dot(x.blocks[1] - x.blocks[2], x.blocks[1] - x.blocks[2])

function grad!(storage, x)
    g = copy(x)
    g.blocks = [x.blocks[1] - x.blocks[2], x.blocks[2] - x.blocks[1]]
    @. storage = g
end

n = 100

lmo1 = FrankWolfe.ScaledBoundLInfNormBall(-ones(n), zeros(n))
lmo2 = FrankWolfe.ProbabilitySimplexOracle(1.0)
prod_lmo = FrankWolfe.ProductLMO((lmo1, lmo2))

x0 = FrankWolfe.BlockVector([-ones(n), [i == 1 ? 1 : 0 for i in 1:n]], [(n,), (n,)], 2 * n)

trajectories = []

# Example for creating a custome block coordinate update order
struct CustomOrder <: FrankWolfe.BlockCoordinateUpdateOrder end

function FrankWolfe.select_update_indices(::CustomOrder, state::FrankWolfe.CallbackState, dual_gaps)
    return [rand() < 1 / n ? 1 : 2 for _ in 1:length(state.lmo.lmos)]
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
display(plot_trajectories(trajectories, labels, xscalelog=true))

# Example for running BCFW with different update methods
trajectories = []

for us in [(FrankWolfe.BPCGStep(), FrankWolfe.FrankWolfeStep()), (FrankWolfe.FrankWolfeStep(), FrankWolfe.BPCGStep()), FrankWolfe.BPCGStep(), FrankWolfe.FrankWolfeStep()]

    _, _, _, _, traj_data = FrankWolfe.block_coordinate_frank_wolfe(
        f,
        grad!,
        prod_lmo,
        x0;
        verbose=true,
        trajectory=true,
        update_step=us,
    )
    push!(trajectories, traj_data)
end

display(plot_trajectories(trajectories, ["BPCG FW", "FW BPCG", "BPCG", "FW"], xscalelog=true))
