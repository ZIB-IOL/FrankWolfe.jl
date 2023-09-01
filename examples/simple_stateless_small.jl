using FrankWolfe
using Plots
using LinearAlgebra
using Random
using Test

include("../examples/plot_utils.jl")

Random.seed!(42)

n = 30

Q = Symmetric(randn(n,n))
e = eigen(Q)
evals = sort!(exp.(2 * randn(n)))
e.values .= evals
const A = Matrix(e)

lmo = FrankWolfe.LpNormLMO{1}(100.0)

x0 = FrankWolfe.compute_extreme_point(lmo, zeros(n));

const b = n * randn(n)

function f(x)
    1/2 * dot(x, A, x) + dot(b, x) - 0.5 * log(sum(x)) + 4000
end

function grad!(storage, x)
    mul!(storage, A, x)
    storage .+= b
    s = sum(x)
    storage .-= 0.5 * inv(s)
end

gradient=collect(x0)

k = 10_000

line_search = FrankWolfe.MonotonicStepSize(x -> sum(x) > 0)
x, v, primal, dual_gap, trajectory_simple = FrankWolfe.frank_wolfe(
    f,
    grad!,
    lmo,
    FrankWolfe.compute_extreme_point(lmo, zeros(n)),
    max_iteration=k,
    line_search=line_search,
    print_iter=k / 10,
    verbose=true,
    gradient=gradient,
    trajectory=true,
);

line_search2 = FrankWolfe.MonotonicGenericStepsize(FrankWolfe.Agnostic(), x -> sum(x) > 0)
x, v, primal, dual_gap, trajectory_restart = FrankWolfe.frank_wolfe(
    f,
    grad!,
    lmo,
    FrankWolfe.compute_extreme_point(lmo, zeros(n)),
    max_iteration=k,
    line_search=line_search2,
    print_iter=k / 10,
    verbose=true,
    gradient=gradient,
    trajectory=true,
);

plot_trajectories([trajectory_simple[1:end], trajectory_restart[1:end]], ["simple", "stateless"], legend_position=:topright)

# simple step iterations about 33% faster

@test line_search.factor == 8

x, v, primal, dual_gap, trajectory_restart_highpres = FrankWolfe.frank_wolfe(
    f,
    grad!,
    lmo,
    FrankWolfe.compute_extreme_point(lmo, zeros(BigFloat, n)),
    max_iteration=10k,
    line_search=line_search2,
    print_iter=k / 10,
    verbose=true,
    gradient=gradient,
    trajectory=true,
);
