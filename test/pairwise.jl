using FrankWolfe
using LinearAlgebra
using Test
using SparseArrays

function test_callback(state, active_set, args...)
    grad0 = similar(state.x)
    state.grad!(grad0, state.x)
    @test grad0 ≈ state.gradient
end

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
        epsilon=3e-7,
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
        epsilon=3e-7,
    )
    @test res_afw[3] ≈ res_bpcg[3]
    @test norm(res_afw[1] - res_bpcg[1]) ≈ 0 atol = 1e-6
    res_bpcg2 = FrankWolfe.blended_pairwise_conditional_gradient(
        f,
        grad!,
        lmo_prob,
        x0,
        max_iteration=6000,
        line_search=FrankWolfe.Adaptive(),
        verbose=false,
        lazy=true,
        epsilon=3e-7,
    )
    @test res_bpcg2[3] ≈ res_bpcg[3] atol = 1e-5
    active_set_afw = res_afw[end]
    storage = copy(active_set_afw.x)
    grad!(storage, active_set_afw.x)
    epsilon = 1e-6
    d = active_set_afw.x * 0
    @inferred FrankWolfe.afw_step(active_set_afw.x, storage, lmo_prob, active_set_afw, epsilon, d)
    @inferred FrankWolfe.lazy_afw_step(
        active_set_afw.x,
        storage,
        lmo_prob,
        active_set_afw,
        epsilon,
        epsilon,
        d,
    )
    res_bpcg = FrankWolfe.blended_pairwise_conditional_gradient(
        f,
        grad!,
        lmo_prob,
        x0,
        max_iteration=6000,
        line_search=FrankWolfe.Adaptive(),
        verbose=false,
        epsilon=3e-7,
        callback=test_callback,
    )
end

@testset "recompute or not last vertex" begin
    n = 10
    f(x) = norm(x)^2
    function grad!(storage, x)
        @. storage = 2x
        return nothing
    end
    lmo_prob = FrankWolfe.ProbabilitySimplexOracle(4)
    lmo = FrankWolfe.TrackingLMO(lmo_prob)
    x0 = FrankWolfe.compute_extreme_point(lmo_prob, ones(10))
    FrankWolfe.blended_pairwise_conditional_gradient(
        f,
        grad!,
        lmo,
        x0,
        max_iteration=6000,
        line_search=FrankWolfe.Adaptive(),
        epsilon=3e-7,
        verbose=false,
    )
    @test lmo.counter == 51
    prev_counter = lmo.counter
    lmo.counter = 0
    FrankWolfe.blended_pairwise_conditional_gradient(
        f,
        grad!,
        lmo,
        x0,
        max_iteration=6000,
        line_search=FrankWolfe.Adaptive(),
        epsilon=3e-7,
        verbose=false,
        recompute_last_vertex=false,
    )
    @test lmo.counter == prev_counter - 1
end
