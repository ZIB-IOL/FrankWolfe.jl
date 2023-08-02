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

dim = 30

m = JuMP.Model(GLPK.Optimizer)
@variable(m, x[1:dim, 1:dim])
@constraint(m, sum(x * ones(dim, dim)) == 2)
@constraint(m, sum(x * I(dim)) <= 2)
@constraint(m, x .>= 0)


lmos = (FrankWolfe.SpectraplexLMO(1.0, dim, true), FrankWolfe.MathOptLMO(m.moi_backend))
x0 = rand(dim, dim)

trajectories = []

for order in instances(FrankWolfe.UpdateOrder)

    _,_,_,_,_,traj_data = FrankWolfe.alternating_linear_minimization(
        FrankWolfe.BCFW(
            update_order=order,
            verbose=true,
            trajectory=true,
        ),
        f,
        grad!,
        lmos,
        x0,
        lambda=1.0,
    );
    push!(trajectories, traj_data)
end

labels = ["full", "cyclic", "stochastic", "adaptive"]

fp = plot_trajectories(trajectories, labels, legend_position=:best, xscalelog=true)

display(fp)
