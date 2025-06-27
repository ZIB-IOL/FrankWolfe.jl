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


lmos = (FrankWolfe.SpectraplexLMO(1.0, dim), FrankWolfe.MathOptLMO(m.moi_backend))
x0 = (zeros(dim, dim), Matrix(I(dim) ./ dim))

trajectories = []

for order in [
    FrankWolfe.FullUpdate(),
    FrankWolfe.CyclicUpdate(),
    FrankWolfe.StochasticUpdate(),
    FrankWolfe.DualGapOrder(),
    FrankWolfe.DualProgressOrder(),
]
    _, _, _, _, _, traj_data = FrankWolfe.alternating_linear_minimization(
        FrankWolfe.block_coordinate_frank_wolfe,
        f,
        grad!,
        lmos,
        x0;
        update_order=order,
        verbose=true,
        trajectory=true,
        update_step=FrankWolfe.BPCGStep(),
    )
    push!(trajectories, traj_data)
end

labels = ["Full", "Cyclic", "Stochastic", "DualGapOrder", "DualProgressOrder"]

println(trajectories[1][1])

fp = plot_trajectories(
    trajectories,
    labels,
    legend_position=:best,
    xscalelog=true,
    reduce_size=true,
    marker_shapes=[:dtriangle, :rect, :circle, :dtriangle, :rect, :circle],
    extra_plot=true,
    extra_plot_label="infeasibility",
)

display(fp)
