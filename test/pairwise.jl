using FrankWolfe
using LinearAlgebra
using Test
using SparseArrays

@testset "Testing Blended Pairwise Conditional Gradients" begin
    f(x) = norm(x)^2
    function grad!(storage, x)
        @. storage = 2x
        return nothing
    end
    lmo_prob = FrankWolfe.ProbabilitySimplexOracle(4)
    x0 = FrankWolfe.compute_extreme_point(lmo_prob, zeros(10))
    res_bpcg = FrankWolfe.blended_pairwise_conditional_gradient(
        f,
        grad!,
        lmo_prob,
        x0,
        max_iteration=6000,
        line_search=FrankWolfe.Adaptive(),
        verbose=false,
    )
    res_afw = FrankWolfe.away_frank_wolfe(
        f,
        grad!,
        lmo_prob,
        x0,
        max_iteration=6000,
        line_search=FrankWolfe.Adaptive(),
        print_iter=100,
        verbose=false,
    )
    @test res_afw[3] ≈ res_bpcg[3]
    @test norm(res_afw[1] - res_bpcg[1]) ≈ 0 atol=1e-6
    res_bpcg2 = FrankWolfe.blended_pairwise_conditional_gradient(
        f,
        grad!,
        lmo_prob,
        x0,
        max_iteration=6000,
        line_search=FrankWolfe.Adaptive(),
        verbose=false,
        lazy=true
    )
    @test res_bpcg2[3] ≈ res_bpcg[3] atol=1e-5
end
