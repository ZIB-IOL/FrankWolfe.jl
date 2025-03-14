using FrankWolfe
using Test
using StableRNGs
using Random
using LinearAlgebra
using SparseArrays

@testset "Corrective steps" begin
    n = 100
    k = 1000
    rng = StableRNG(42)
    Random.seed!(rng, 42)

    # Create quadratic function over simplex
    xpi = rand(rng, n)
    total = sum(xpi)
    xp = xpi ./ total

    f(x) = norm(x - xp)^2
    function grad!(storage, x)
        @. storage = 2 * (x - xp)
    end

    lmo = FrankWolfe.ProbabilitySimplexOracle(1.0)
    x0 = FrankWolfe.compute_extreme_point(lmo, zeros(n))

    x_afw, v_afw, primal_afw, dual_gap_afw, traj_afw = FrankWolfe.away_frank_wolfe(
        f,
        grad!,
        lmo,
        copy(x0),
        max_iteration=k,
        verbose=true,
        print_iter=1,
        lazy=true
    )

    for lazy in (true, false)
        x_cfw, v_cfw, primal_cfw, dual_gap_cfw, traj_cfw = FrankWolfe.corrective_frankwolfe(
            f,
            grad!,
            lmo,
            FrankWolfe.AwayStep(lazy),
            FrankWolfe.ActiveSet([(1.0, copy(x0))]),
            max_iteration=k,
            verbose=false,
        )
        @test abs(primal_cfw - primal_afw) <= 1e-6
        @test isapprox(x_cfw, x_afw, rtol=1e-5)

        x_bp, _, primal_bp, dual_gap_bp, _ = FrankWolfe.corrective_frankwolfe(
            f,
            grad!,
            lmo,
            FrankWolfe.BlendedPairwiseStep(lazy),
            FrankWolfe.ActiveSet([(1.0, copy(x0))]),
            max_iteration=k,
            verbose=false,
        )
        # Check solutions match
        @test abs(primal_bp - primal_afw) <= 1e-6
        @test isapprox(x_bp, x_afw, rtol=1e-5)
    end

    lazy = true
    x_hyb, _, primal_hyb, _ = FrankWolfe.corrective_frankwolfe(
        f,
        grad!,
        lmo,
        FrankWolfe.HybridPairAwayStep(lazy, copy(x0)),
        FrankWolfe.ActiveSet([(1.0, copy(x0))]),
        max_iteration=k,
        verbose=false,
    )
    @test abs(primal_hyb - primal_afw) <= 1e-6
    @test isapprox(x_hyb, x_afw, rtol=1e-5)    
end
