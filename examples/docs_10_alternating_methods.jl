# # Alternating methods

# In this example we will compare [`FrankWolfe.alternating_linear_minimization`](@ref) and [`FrankWolfe.alternating_projections`](@ref) for a very simple feasibility problem.

# We consider the probability simplex
# ```math
# P = \{ x \in \mathbb{R}^n \colon \sum_{i=1}^n x_i = 1, x_i \geq 0 ~~ i=1,\dots,n\} ~.
# ```
# and a scaled, shifted ``\ell^{\infty}`` norm ball
# ```math
# Q = [-1,0]^n ~.
# ```
# The goal is to find either a point in the intersection, `` x \in P \cap Q``, or a pair of points, ``(x_P, x_Q) \in P \times Q``, which attains minimal distance between ``P`` and ``Q``,
# ```math
# \|x_P - x_Q\|_2 = \min_{(x,y) \in P \times Q} \|x - y \|_2 ~.
# ```

using FrankWolfe
include("../examples/plot_utils.jl")

# ## Setting up objective, gradient and linear minimization oracles
# Since we only consider the feasibility problem the objective function as well as the gradient are zero.

n = 20

f(x) = 0

function grad!(storage, x)
    @. storage = zero(x)
end


lmo1 = FrankWolfe.ProbabilitySimplexOracle(1.0)
lmo2 = FrankWolfe.ScaledBoundLInfNormBall(-ones(n), zeros(n))
lmos = (lmo1, lmo2)

x0 = rand(n)

target_tolerance = 1e-6

trajectories = [];

# ## Running Alternating Linear Minimization
# We run Alternating Linear Minimization (ALM) with [`FrankWolfe.block_coordinate_frank_wolfe`](@ref).
# This method allows three different update orders, `FullUpdate`, `CyclicUpdate` and `Stochasticupdate`.
# Accordingly both blocks are updated either simulatenously, sequentially or in random order.

for order in [FrankWolfe.FullUpdate(), FrankWolfe.CyclicUpdate(), FrankWolfe.StochasticUpdate()]

    _, _, _, _, _, alm_trajectory = FrankWolfe.alternating_linear_minimization(
        FrankWolfe.block_coordinate_frank_wolfe,
        f,
        grad!,
        lmos,
        x0,
        update_order=order,
        verbose=true,
        trajectory=true,
        epsilon=target_tolerance,
    )
    push!(trajectories, alm_trajectory)
end

# As an alternative to Block-Coordiante Frank-Wolfe (BCFW), one can also run alternating linear minimization with standard Frank-Wolfe algorithm.
# These methods perform then the full (simulatenous) update at each iteration. In this example we also use [`FrankWolfe.away_frank_wolfe`](@ref).

_, _, _, _, _, afw_trajectory = FrankWolfe.alternating_linear_minimization(
    FrankWolfe.away_frank_wolfe,
    f,
    grad!,
    lmos,
    x0,
    verbose=true,
    trajectory=true,
    epsilon=target_tolerance,
)
push!(trajectories, afw_trajectory);

# ## Running Alternating Projections
# Unlike ALM, Alternating Projections (AP) is only suitable for feasibility problems. One omits the objective and gradient as parameters.
_, _, _, _, ap_trajectory = FrankWolfe.alternating_projections(
    lmos,
    x0,
    trajectory=true,
    verbose=true,
    print_iter=100,
    epsilon=target_tolerance,
)
push!(trajectories, ap_trajectory);

# ## Plotting the resulting trajectories

labels = ["BCFW - Full", "BCFW - Cyclic", "BCFW - Stochastic", "AFW", "AP"]

plot_trajectories(trajectories, labels, xscalelog=true)
