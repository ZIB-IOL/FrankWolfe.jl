using FrankWolfe
using LinearAlgebra
using Test
using SparseArrays

function test_callback(state, active_set, args...)
    grad0 = similar(state.x)
    state.grad!(grad0, state.x)
    @test grad0 ≈ state.gradient
end

@testset "Testing active set Frank-Wolfe variants, BPFW, AFW, and PFW, including their lazy versions" begin
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
        line_search=FrankWolfe.AdaptiveZerothOrder(),
        verbose=false,
        epsilon=3e-7,
    )
    res_bpcg_lazy = FrankWolfe.blended_pairwise_conditional_gradient(
        f,
        grad!,
        lmo_prob,
        x0,
        max_iteration=6000,
        line_search=FrankWolfe.AdaptiveZerothOrder(),
        verbose=false,
        epsilon=3e-7,
        lazy=true,
    )
    res_afw = FrankWolfe.away_frank_wolfe(
        f,
        grad!,
        lmo_prob,
        x0,
        max_iteration=6000,
        line_search=FrankWolfe.AdaptiveZerothOrder(),
        print_iter=100,
        verbose=false,
        epsilon=3e-7,
    )
    res_afw_lazy = FrankWolfe.away_frank_wolfe(
        f,
        grad!,
        lmo_prob,
        x0,
        max_iteration=6000,
        line_search=FrankWolfe.AdaptiveZerothOrder(),
        print_iter=100,
        verbose=false,
        epsilon=3e-7,
        lazy=true,
    )
    res_pfw = FrankWolfe.pairwise_frank_wolfe(
        f,
        grad!,
        lmo_prob,
        x0,
        max_iteration=6000,
        line_search=FrankWolfe.AdaptiveZerothOrder(),
        print_iter=100,
        verbose=false,
        epsilon=3e-7,
    )
    res_pfw_lazy = FrankWolfe.pairwise_frank_wolfe(
        f,
        grad!,
        lmo_prob,
        x0,
        max_iteration=6000,
        line_search=FrankWolfe.AdaptiveZerothOrder(),
        print_iter=100,
        verbose=false,
        lazy=true,
        epsilon=3e-7,
    )
    @test res_afw[3] ≈ res_bpcg[3]
    @test res_afw[3] ≈ res_pfw[3]
    @test res_afw[3] ≈ res_afw_lazy[3]
    @test res_pfw[3] ≈ res_pfw_lazy[3]
    @test res_bpcg[3] ≈ res_bpcg_lazy[3]
    @test norm(res_afw[1] - res_bpcg[1]) ≈ 0 atol = 1e-6
    @test norm(res_afw[1] - res_pfw[1]) ≈ 0 atol = 1e-6
    @test norm(res_afw[1] - res_afw_lazy[1]) ≈ 0 atol = 1e-6
    @test norm(res_pfw[1] - res_pfw_lazy[1]) ≈ 0 atol = 1e-6
    @test norm(res_bpcg[1] - res_bpcg_lazy[1]) ≈ 0 atol = 1e-6
    res_bpcg2 = FrankWolfe.blended_pairwise_conditional_gradient(
        f,
        grad!,
        lmo_prob,
        x0,
        max_iteration=6000,
        line_search=FrankWolfe.AdaptiveZerothOrder(),
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
        line_search=FrankWolfe.AdaptiveZerothOrder(),
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
        line_search=FrankWolfe.AdaptiveZerothOrder(),
        epsilon=3e-7,
        verbose=false,
    )
    @test lmo.counter <= 51
    prev_counter = lmo.counter
    lmo.counter = 0
    FrankWolfe.blended_pairwise_conditional_gradient(
        f,
        grad!,
        lmo,
        x0,
        max_iteration=6000,
        line_search=FrankWolfe.AdaptiveZerothOrder(),
        epsilon=3e-7,
        verbose=false,
        recompute_last_vertex=false,
    )
    @test lmo.counter == prev_counter - 1
end

@testset "Quadratic active set" begin
    Random.seed!(0)
    n = 3 # number of dimensions
    p = 10^2 # number of points
    function simple_reg_loss(θ, data_point)
        (xi, yi) = data_point
        (a, b) = (θ[1:end-1], θ[end])
        pred = a ⋅ xi + b
        return (pred - yi)^2 / 2
    end
    function ∇simple_reg_loss(storage, θ, data_point)
        (xi, yi) = data_point
        (a, b) = (θ[1:end-1], θ[end])
        pred = a ⋅ xi + b
        @. storage[1:end-1] += xi * (pred - yi)
        storage[end] += pred - yi
        return storage
    end
    xs = [10randn(n) for _ in 1:p]
    bias = 4
    params_perfect = [1:n; bias]
    data_noisy = [(x, x ⋅ (1:n) + bias + 0.5 * randn()) for x in xs]
    f(x) = sum(simple_reg_loss(x, data_point) for data_point in data_noisy)
    function gradf(storage, x)
        storage .= 0
        for dp in data_noisy
            ∇simple_reg_loss(storage, x, dp)
        end
    end
    lmo = FrankWolfe.LpNormLMO{Float64, 2}(1.05 * norm(params_perfect))
    x0 = FrankWolfe.compute_extreme_point(lmo, zeros(Float64, n+1))
    active_set = FrankWolfe.ActiveSetQuadratic([(1.0, x0)], gradf)
    res = FrankWolfe.blended_pairwise_conditional_gradient(
        f,
        gradf,
        lmo,
        active_set;
        verbose=false,
        lazy=true,
        line_search=FrankWolfe.Adaptive(L_est=10.0, relaxed_smoothness=true),
        trajectory=true,
    )
    @test abs(res[3] - 12.084) ≤ 0.01
end
