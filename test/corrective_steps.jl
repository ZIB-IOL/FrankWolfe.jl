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

    x_afw, v_afw, primal_afw, dual_gap_afw, status, traj_afw = FrankWolfe.away_frank_wolfe(
        f,
        grad!,
        lmo,
        copy(x0),
        max_iteration=k,
        verbose=true,
        print_iter=1,
        lazy=true,
    )

    for lazy in (true, false)
        x_cfw, v_cfw, primal_cfw, dual_gap_cfw, status_aw, traj_cfw =
            FrankWolfe.corrective_frank_wolfe(
                f,
                grad!,
                lmo,
                FrankWolfe.AwayStep(lazy),
                FrankWolfe.ActiveSet([(1.0, x0)]),
                max_iteration=k,
                verbose=false,
            )
        @test abs(primal_cfw - primal_afw) <= 1e-6
        @test isapprox(x_cfw, x_afw, rtol=1e-5)
        @test status_aw == FrankWolfe.STATUS_OPTIMAL

        x_bp, _, primal_bp, dual_gap_bp, status_bp, _ = FrankWolfe.corrective_frank_wolfe(
            f,
            grad!,
            lmo,
            FrankWolfe.BlendedPairwiseStep(lazy),
            FrankWolfe.ActiveSet([(1.0, x0)]),
            max_iteration=k,
            verbose=false,
        )
        # Check solutions match
        @test abs(primal_bp - primal_afw) <= 1e-6
        @test status_bp == FrankWolfe.STATUS_OPTIMAL
        @test isapprox(x_bp, x_afw, rtol=1e-5)
        x_pw, _, primal_pw, dualgap_pw, status_pw, _ = FrankWolfe.corrective_frank_wolfe(
            f,
            grad!,
            lmo,
            FrankWolfe.PairwiseStep(lazy),
            FrankWolfe.ActiveSet([(1.0, x0)]),
            max_iteration=k,
            verbose=false,
        )
        x_pw0, _, primal_pw0, dualgap_pw0, _ = FrankWolfe.pairwise_frank_wolfe(
            f,
            grad!,
            lmo,
            FrankWolfe.ActiveSet([(1.0, x0)]),
            max_iteration=k,
            verbose=false,
            lazy=lazy,
        )

        @test abs(primal_pw - primal_pw0) <= 1e-6
        @test isapprox(x_pw, x_pw0, rtol=1e-5)
        @test status_pw == FrankWolfe.STATUS_OPTIMAL
    end

    lazy = true
    x_hyb, _, primal_hyb, _, status_h, _, _ = FrankWolfe.corrective_frank_wolfe(
        f,
        grad!,
        lmo,
        FrankWolfe.HybridPairAwayStep(lazy, copy(x0)),
        FrankWolfe.ActiveSet([(1.0, x0)]),
        max_iteration=k,
        verbose=false,
    )
    @test abs(primal_hyb - primal_afw) <= 1e-6
    @test isapprox(x_hyb, x_afw, rtol=1e-5)
    @test status_h == FrankWolfe.STATUS_OPTIMAL
end
