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
        weight_purge_threshold=1e-12,
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
    n = 2 # number of dimensions
    p = 10 # number of points
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
    xs = [[9.42970533446119, 1.3392275765318449],
          [15.250689085124804, 1.2390123120559722],
          [-12.057722842599361, 3.1181717536024807],
          [-2.3464126496126, -10.873522937105326],
          [4.623106804313759, -0.8059308663320504],
          [-8.124306879044243, -20.610343848003204],
          [3.1305636922867732, -4.794303128671186],
          [-9.443890241279835, 18.243232066781086],
          [-10.582972181702795, 2.9216495153528084],
          [12.469122418416605, -4.2927539788825735]]
    params_perfect = [1:n; 4]
    data_noisy = [([9.42970533446119, 1.3392275765318449], 16.579645754247938),
                  ([15.250689085124804, 1.2390123120559722], 21.79567508806334),
                  ([-12.057722842599361, 3.1181717536024807], -1.0588448811381594),
                  ([-2.3464126496126, -10.873522937105326], -20.03150790822045),
                  ([4.623106804313759, -0.8059308663320504], 6.408358929519689),
                  ([-8.124306879044243, -20.610343848003204], -45.189085987370525),
                  ([3.1305636922867732, -4.794303128671186], -2.5753631975362286),
                  ([-9.443890241279835, 18.243232066781086], 30.498897745427072),
                  ([-10.582972181702795, 2.9216495153528084], -0.5085178107814902),
                  ([12.469122418416605, -4.2927539788825735], 7.843317917334855)]
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
    @test abs(res[3] - 0.70939) ≤ 0.001
    @test res[4] ≤ 1e-2
end
