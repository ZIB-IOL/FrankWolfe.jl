using Test
using FrankWolfe
using LinearAlgebra
using DoubleFloats
using Random
using StableRNGs

rng = StableRNG(42)
Random.seed!(rng, 42)

@testset "Multi-precision tests" begin
    rhs = 1
    n = 100
    k = 1000

    xp = zeros(n)

    L = 2
    bound = 2 * L * 2 / (k + 2)

    f(x) = norm(x - xp)^2
    function grad!(storage, x)
        @. storage = 2 * (x - xp)
    end
    test_types = (Float16, Float32, Float64, Double64, BigFloat, Rational{BigInt})

    @testset "Multi-precision test for $T" for T in test_types
        lmo = FrankWolfe.ProbabilitySimplexOracle{T}(rhs)
        direction = rand(rng, n)
        x0 = FrankWolfe.compute_extreme_point(lmo, direction)

        x, v, primal, dual_gap, trajectory = FrankWolfe.frank_wolfe(
            f,
            grad!,
            lmo,
            x0,
            max_iteration=k,
            line_search=FrankWolfe.Agnostic(),
            print_iter=k / 10,
            memory_mode=FrankWolfe.OutplaceEmphasis(),
            verbose=false,
        )

        @test eltype(x0) == T
        @test primal - 1 / n <= bound

        x, v, primal, dual_gap, trajectory = FrankWolfe.frank_wolfe(
            f,
            grad!,
            lmo,
            x0,
            max_iteration=k,
            line_search=FrankWolfe.Agnostic(),
            print_iter=k / 10,
            memory_mode=FrankWolfe.InplaceEmphasis(),
            verbose=false,
        )

        @test eltype(x0) == T
        @test primal - 1 // n <= bound

        x, v, primal, dual_gap, trajectory = FrankWolfe.away_frank_wolfe(
            f,
            grad!,
            lmo,
            x0,
            max_iteration=k,
            line_search=FrankWolfe.AdaptiveZerothOrder(),
            print_iter=k / 10,
            memory_mode=FrankWolfe.InplaceEmphasis(),
            verbose=false,
        )

        @test eltype(x0) == T
        @test primal - 1 // n <= bound

        x, v, primal, dual_gap, trajectory, _ = FrankWolfe.blended_conditional_gradient(
            f,
            grad!,
            lmo,
            x0,
            max_iteration=k,
            line_search=FrankWolfe.AdaptiveZerothOrder(),
            print_iter=k / 10,
            memory_mode=FrankWolfe.InplaceEmphasis(),
            verbose=false,
        )

        @test eltype(x0) == T
        @test primal - 1 // n <= bound

    end
end
