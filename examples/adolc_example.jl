using FrankWolfe
using ADOLC
using LinearAlgebra
using Test

## Data setup
n = Int(100)

xpi = rand(n);
total = sum(xpi);
const xp = xpi ./ total;

f(x) = norm(x - xp)^2

println("Automatic differentiation")

const tape_id = 1
ADOLC.derivative(f, zeros(n), :jac, tape_id=tape_id)
c = CxxVector(zeros(n))

function grad!(storage, x)
 ADOLC.gradient!(c, f, n, x, tape_id, true)   # derivate in the direction y
 @. storage = c
end

lmo_radius = 2.5
lmo = FrankWolfe.FrankWolfe.ProbabilitySimplexOracle(lmo_radius)

x00 = FrankWolfe.compute_extreme_point(lmo, zeros(n))
gradient = collect(x00)

x_au, _, primal_au, dual_gap_au, _ = FrankWolfe.frank_wolfe(
    f,
    grad!,
    lmo,
    collect(copy(x00)),
    line_search=FrankWolfe.Adaptive(),
    memory_mode=FrankWolfe.InplaceEmphasis(),
    verbose=true,
    trajectory=false,
);

println("\nHandwritten Gradient")

function grad!(storage, x)
    @. storage = 2 * (x - xp)
    return nothing
end

lmo_radius = 2.5
lmo = FrankWolfe.FrankWolfe.ProbabilitySimplexOracle(lmo_radius)

x00 = FrankWolfe.compute_extreme_point(lmo, zeros(n))
gradient = collect(x00)

x, _, primal, dual_gap, _ = FrankWolfe.frank_wolfe(
    f,
    grad!,
    lmo,
    collect(copy(x00)),
    line_search=FrankWolfe.Adaptive(),
    memory_mode=FrankWolfe.InplaceEmphasis(),
    verbose=true,
    trajectory=false,
);

@testset "Automatic differentiation" begin
    @test primal >= primal_au - dual_gap_au
    @test primal_au >= primal - dual_gap
end