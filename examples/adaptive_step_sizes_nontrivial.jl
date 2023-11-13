using FrankWolfe
using ProgressMeter
using Random
using Plots

using LinearAlgebra

include("../examples/plot_utils.jl")

##############
# example to demonstrate additional numerical stability of first order adaptive line search
# try different sizes of n
##############

# n = Int(1e2)
# n = Int(3e2)
n = Int(5e2)
k = Int(1e3)

######

seed = 10
Random.seed!(seed)


const A = let
    A = randn(n, n)
    A' * A
end

@assert isposdef(A) == true

const y = Random.rand(Bool, n) * 0.6 .+ 0.3

function f(x)
    d = x - y
    return dot(d, A, d)
end

function grad!(storage, x)
    mul!(storage, A, x)
    return mul!(storage, A, y, -2, 2)
end

# lmo = FrankWolfe.KSparseLMO(40, 1.0);
lmo = FrankWolfe.UnitSimplexOracle(1.0);
# lmo = FrankWolfe.ScaledBoundLInfNormBall(zeros(n),ones(n))

x00 = FrankWolfe.compute_extreme_point(lmo, zeros(n));

FrankWolfe.benchmark_oracles(x -> f(x), (str, x) -> grad!(str, x), () -> randn(n), lmo; k=100)

println("\n==> Adaptive (1-order) if you do not know L.\n")

x0 = deepcopy(x00)

@time x, v, primal, dual_gap, trajectory_adaptive_fo = FrankWolfe.frank_wolfe(
    f,
    grad!,
    lmo,
    x0,
    max_iteration=k,
    line_search=FrankWolfe.Adaptive(),
    print_iter=k / 10,
    memory_mode=FrankWolfe.InplaceEmphasis(),
    verbose=true,
    trajectory=true,
);

println("\n==> Adaptive (0-order) if you do not know L.\n")

x0 = deepcopy(x00)

@time x, v, primal, dual_gap, trajectory_adaptive_zo = FrankWolfe.frank_wolfe(
    f,
    grad!,
    lmo,
    x0,
    max_iteration=k,
    line_search=FrankWolfe.AdaptiveZerothOrder(),
    print_iter=k / 10,
    memory_mode=FrankWolfe.InplaceEmphasis(),
    verbose=true,
    trajectory=true,
);

data = [trajectory_adaptive_zo, trajectory_adaptive_fo]
label = ["adaptive 0-order", "adaptive 1-order"]


plot_trajectories(data, label, xscalelog=true)
