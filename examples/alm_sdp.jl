using FrankWolfe
using LinearAlgebra
using JuMP
const MOI = JuMP.MOI
import GLPK

include("../examples/plot_utils.jl")


f(x) = 0.0

function grad!(storage, x)
    @. storage = 0
end

dim = 30

m = JuMP.Model(GLPK.Optimizer)
@variable(m, x[1:dim, 1:dim])
@constraint(m, sum(x * ones(dim, dim)) == 2)
@constraint(m, sum(x * I(dim)) <= 2)
@constraint(m, x .>= 0)


lmos = (FrankWolfe.SpectraplexLMO(1.0, dim, true), FrankWolfe.MathOptLMO(m.moi_backend))
x0 = rand(dim, dim)

trajectories = []

for order in [FrankWolfe.FullUpdate(), FrankWolfe.CyclicUpdate(), FrankWolfe.StochasticUpdate()]

    _, _, _, _, _, traj_data = FrankWolfe.alternating_linear_minimization(
        FrankWolfe.block_coordinate_frank_wolfe,
        f,
        grad!,
        lmos,
        x0;
        lambda=1.0,
        update_order=order,
        line_search=FrankWolfe.Adaptive(relaxed_smoothness=true),
        verbose=true,
        trajectory=true,
        max_iteration=10000,
    )
    push!(trajectories, traj_data)
end

labels = ["Full", "Cyclic", "Stochastic"]

fp = plot_trajectories(trajectories, labels, legend_position=:best, xscalelog=true, reduce_size=true, marker_shapes=[:dtriangle, :rect, :cross])

display(fp)
