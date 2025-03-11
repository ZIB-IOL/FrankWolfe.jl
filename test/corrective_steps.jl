using FrankWolfe
using Test
using StableRNGs
using Random
using LinearAlgebra

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
    active_set = FrankWolfe.ActiveSet([(1.0, copy(x0))])

    # Run corrective FW with away steps
    x_cfw, v_cfw, primal_cfw, dual_gap_cfw, traj_cfw = FrankWolfe.corrective_frankwolfe(
        f,
        grad!,
        lmo,
        FrankWolfe.AwayStep(),
        active_set,
        max_iteration=k,
        verbose=false
    )

    # Run away FW
    x_afw, v_afw, primal_afw, dual_gap_afw, traj_afw = FrankWolfe.away_frank_wolfe(
        f,
        grad!,
        lmo,
        copy(x0),
        max_iteration=k,
        verbose=false
    )

    # Check solutions match
    @test abs(primal_cfw - primal_afw) <= 1e-6
    @test isapprox(x_cfw, x_afw, rtol=1e-6)
end
