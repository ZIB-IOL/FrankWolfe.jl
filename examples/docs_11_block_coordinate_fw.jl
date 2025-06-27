# # Block-Coordinate Frank-Wolfe and Block-Vectors

# In this example, we demonstrate the usage of the [`FrankWolfe.block_coordinate_frank_wolfe`](@ref) and [`FrankWolfe.BlockVector`](@ref).
# We consider the problem of minimizing the squared Euclidean distance between two sets.
# We compare different update orders and different update steps.

# ## Import and setup
# We first import the necessary packages and include the code for plotting the results.
using FrankWolfe
using LinearAlgebra

include("plot_utils.jl")

# Next, we define the objective function and its gradient. The iterates `x` are instances of the [`FrankWolfe.BlockVector`](@ref) type.
# The different blocks of the vector can be accessed via the `blocks` field.

f(x) = dot(x.blocks[1] - x.blocks[2], x.blocks[1] - x.blocks[2])

function grad!(storage, x)
    @. storage.blocks = [x.blocks[1] - x.blocks[2], x.blocks[2] - x.blocks[1]]
end

# In our example we consider the probability simplex and an L-infinity norm ball as the feasible sets.
n = 100
lmo1 = FrankWolfe.ScaledBoundLInfNormBall(-ones(n), zeros(n))
lmo2 = FrankWolfe.ProbabilitySimplexOracle(1.0)
prod_lmo = FrankWolfe.ProductLMO((lmo1, lmo2))

# We initialize the starting point `x0` as a [`FrankWolfe.BlockVector`](@ref) with two blocks.
# The two other arguments are the block sizes and the overall number of entries.
x0 = FrankWolfe.BlockVector([-ones(n), [i == 1 ? 1 : 0 for i in 1:n]], [(n,), (n,)], 2 * n);


# ## Running block-coordinate Frank-Wolfe with different update-orders

# In a first step, we compare different update orders. There are three different update orders implemented, 
# [`FrankWolfe.FullUpdate`](@ref), [`CyclicUpdate`](@ref) and [`Stochasticupdate`](@ref).
# For creating a custome [`FrankWolfe.BlockCoordinateUpdateOrder`](@ref), one needs to implement the function `select_update_indices`.
struct CustomOrder <: FrankWolfe.BlockCoordinateUpdateOrder end

function FrankWolfe.select_update_indices(::CustomOrder, state::FrankWolfe.CallbackState, dual_gaps)
    return [rand() < 1 / n ? 1 : 2 for _ in 1:length(state.lmo.lmos)]
end

# We run the block-coordinate Frank-Wolfe method with the different update orders and store the trajectories.

trajectories = []

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
# ### Plotting the results
labels = ["Full update", "Cyclic order", "Stochstic order", "Custom order"]
plot_trajectories(trajectories, labels, xscalelog=true)

# ## Running BCFW with different update methods
# As a second step, we compare different update steps. We consider the [`FrankWolfe.BPCGStep`](@ref) and the [`FrankWolfe.FrankWolfeStep`](@ref).
# One can either pass a tuple of [`FrankWolfe.UpdateStep`](@ref) to define for each block the update procedure or pass a single update step so that each block uses the same procedure.

trajectories = []

for us in [
    (FrankWolfe.BPCGStep(), FrankWolfe.FrankWolfeStep()),
    (FrankWolfe.FrankWolfeStep(), FrankWolfe.BPCGStep()),
    FrankWolfe.BPCGStep(),
    FrankWolfe.FrankWolfeStep(),
]

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
# ### Plotting the results
labels = ["BPCG FW", "FW BPCG", "BPCG", "FW"]
plot_trajectories(trajectories, labels, xscalelog=true)
