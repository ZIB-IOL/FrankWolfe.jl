using FrankWolfe
using LinearAlgebra
using JuMP
const MOI = JuMP.MOI
import GLPK

include("../examples/plot_utils.jl")


f(x) = 0.0

function grad!(storage, x)
    @. storage = zero(x)
end

dim = 10

m = JuMP.Model(GLPK.Optimizer)
@variable(m, x[1:dim, 1:dim])
@constraint(m, sum(x * ones(dim, dim)) == 2)
@constraint(m, sum(x * I(dim)) <= 2)
@constraint(m, x .>= 0)


lmos = (FrankWolfe.SpectraplexLMO(1.0, dim, true, 1000), FrankWolfe.MathOptLMO(m.moi_backend))
x0 = rand(dim, dim)

trajectories = []

for order in [FrankWolfe.FullUpdate(), FrankWolfe.CyclicUpdate(), FrankWolfe.StochasticUpdate()]
    for step in [FrankWolfe.FrankWolfeStep(), FrankWolfe.BPCGStep()]
        _, _, _, _, _, traj_data = FrankWolfe.alternating_linear_minimization(
            FrankWolfe.block_coordinate_frank_wolfe,
            f,
            grad!,
            lmos,
            x0;
            update_order=order,
            #line_search=FrankWolfe.Adaptive(relaxed_smoothness=true),
            verbose=true,
            trajectory=true,
            update_step=step,
        )
        push!(trajectories, traj_data)
    end
end

labels = ["Full - Vanilla", "Full - BPCG", "Cyclic - Vanilla", "Cyclic - BPCG", "Stochastic - Vanilla", "Stochastic - BPCG"]

fp = plot_trajectories(
    trajectories,
    labels,
    legend_position=:best,
    xscalelog=true,
    reduce_size=true,
    marker_shapes=[:dtriangle, :rect, :circle, :dtriangle, :rect, :circle],
)

display(fp)
