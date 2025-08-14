using Test
using FrankWolfe
using LinearAlgebra
using Random
using StableRNGs

rng = StableRNG(42)
Random.seed!(rng, 42)

@testset "Testing memory_mode blas vs memory" begin
    n = Int(1e5)
    k = 100
    xpi = rand(rng, n)
    total = sum(xpi)
    xp = xpi ./ total
    f(x) = norm(x - xp)^2
    function grad!(storage, x)
        @. storage = 2 * (x - xp)
        return nothing
    end
    @testset "Using sparse structure" begin
        lmo_prob = FrankWolfe.ProbabilitySimplexOracle(1.0)
        x0 = FrankWolfe.compute_extreme_point(lmo_prob, spzeros(n))

        x, v, primal, dual_gap, status, trajectory = FrankWolfe.frank_wolfe(
            f,
            grad!,
            lmo_prob,
            x0,
            max_iteration=k,
            line_search=FrankWolfe.Backtracking(),
            print_iter=k / 10,
            verbose=false,
            memory_mode=FrankWolfe.OutplaceEmphasis(),
        )

        @test primal < f(x0)

        x, v, primal, dual_gap, status, trajectory = FrankWolfe.frank_wolfe(
            f,
            grad!,
            lmo_prob,
            x0,
            max_iteration=k,
            line_search=FrankWolfe.Backtracking(),
            print_iter=k / 10,
            verbose=false,
            memory_mode=FrankWolfe.InplaceEmphasis(),
        )
        @test primal < f(x0)
        x, v, primal, dual_gap, status, trajectory = FrankWolfe.frank_wolfe(
            f,
            grad!,
            lmo_prob,
            x0,
            max_iteration=k,
            line_search=FrankWolfe.MonotonicStepSize(),
            print_iter=k / 10,
            verbose=false,
            memory_mode=FrankWolfe.InplaceEmphasis(),
        )
        @test primal < f(x0)
        x, v, primal, dual_gap, status, trajectory = FrankWolfe.frank_wolfe(
            f,
            grad!,
            lmo_prob,
            x0,
            max_iteration=k,
            line_search=FrankWolfe.MonotonicNonConvexStepSize(),
            print_iter=k / 10,
            verbose=false,
            memory_mode=FrankWolfe.InplaceEmphasis(),
        )
        @test primal < f(x0)
    end
end
