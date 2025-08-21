using FrankWolfe
using Test
using LinearAlgebra
using Random
using StableRNGs

rng = StableRNG(42)
Random.seed!(rng, 42)

@testset "Testing vanilla Frank-Wolfe" begin
    f(x) = norm(x)^2
    function grad!(storage, x)
        return storage .= 2x
    end
    lmo_prob = FrankWolfe.ProbabilitySimplexOracle(1)
    x0 = FrankWolfe.compute_extreme_point(lmo_prob, zeros(5))
    @test abs(
        FrankWolfe.frank_wolfe(
            f,
            grad!,
            lmo_prob,
            x0,
            max_iteration=10000,
            line_search=FrankWolfe.GeneralizedAgnostic(),
            verbose=false,
        )[3] - 0.2,
    ) < 1.0e-5
    g(x) = 2 + log(1 + log(x + 1))
    @test abs(
        FrankWolfe.frank_wolfe(
            f,
            grad!,
            lmo_prob,
            x0,
            max_iteration=10000,
            line_search=FrankWolfe.GeneralizedAgnostic(g),
            verbose=false,
        )[3] - 0.2,
    ) < 1.0e-5
    @test abs(
        FrankWolfe.frank_wolfe(
            f,
            grad!,
            lmo_prob,
            x0,
            max_iteration=1000,
            line_search=FrankWolfe.Agnostic(),
            verbose=false,
        )[3] - 0.2,
    ) < 1.0e-5
    @test abs(
        FrankWolfe.frank_wolfe(
            f,
            grad!,
            lmo_prob,
            x0,
            max_iteration=1000,
            line_search=FrankWolfe.Agnostic(),
            verbose=false,
            gradient=collect(similar(x0)),
        )[3] - 0.2,
    ) < 1.0e-5
    @test abs(
        FrankWolfe.frank_wolfe(
            f,
            grad!,
            lmo_prob,
            x0,
            max_iteration=1000,
            line_search=FrankWolfe.Goldenratio(),
            verbose=false,
        )[3] - 0.2,
    ) < 1.0e-5
    @test abs(
        FrankWolfe.frank_wolfe(
            f,
            grad!,
            lmo_prob,
            x0,
            max_iteration=1000,
            line_search=FrankWolfe.Backtracking(),
            verbose=false,
        )[3] - 0.2,
    ) < 1.0e-5
    @test abs(
        FrankWolfe.frank_wolfe(
            f,
            grad!,
            lmo_prob,
            x0,
            max_iteration=1000,
            line_search=FrankWolfe.Nonconvex(),
            verbose=false,
        )[3] - 0.2,
    ) < 1.0e-2
    @test FrankWolfe.frank_wolfe(
        f,
        grad!,
        lmo_prob,
        x0,
        max_iteration=1000,
        line_search=FrankWolfe.Shortstep(2.0),
        verbose=false,
    )[3] ≈ 0.2
    @test abs(
        FrankWolfe.frank_wolfe(
            f,
            grad!,
            lmo_prob,
            x0,
            max_iteration=1000,
            line_search=FrankWolfe.Nonconvex(),
            verbose=false,
        )[3] - 0.2,
    ) < 1.0e-2
    @test abs(
        FrankWolfe.frank_wolfe(
            f,
            grad!,
            lmo_prob,
            x0,
            max_iteration=1000,
            line_search=FrankWolfe.Agnostic(),
            verbose=false,
            momentum=0.9,
        )[3] - 0.2,
    ) < 1.0e-3
    @test abs(
        FrankWolfe.frank_wolfe(
            f,
            grad!,
            lmo_prob,
            x0,
            max_iteration=1000,
            line_search=FrankWolfe.Agnostic(),
            verbose=false,
            momentum=0.5,
        )[3] - 0.2,
    ) < 1.0e-3
    @test abs(
        FrankWolfe.frank_wolfe(
            f,
            grad!,
            lmo_prob,
            x0,
            max_iteration=1000,
            line_search=FrankWolfe.Agnostic(),
            verbose=false,
            momentum=0.9,
            memory_mode=FrankWolfe.InplaceEmphasis(),
        )[3] - 0.2,
    ) < 1.0e-3
end

@testset "Testing Lazified Conditional Gradients" begin
    f(x) = norm(x)^2
    function grad!(storage, x)
        @. storage = 2x
        return nothing
    end
    lmo_prob = FrankWolfe.ProbabilitySimplexOracle(1)
    x0 = FrankWolfe.compute_extreme_point(lmo_prob, zeros(5))
    @test abs(
        FrankWolfe.lazified_conditional_gradient(
            f,
            grad!,
            lmo_prob,
            x0,
            max_iteration=1000,
            line_search=FrankWolfe.Goldenratio(),
            verbose=false,
        )[3] - 0.2,
    ) < 1.0e-5
    @test abs(
        FrankWolfe.lazified_conditional_gradient(
            f,
            grad!,
            lmo_prob,
            x0,
            max_iteration=1000,
            line_search=FrankWolfe.Backtracking(),
            verbose=false,
        )[3] - 0.2,
    ) < 1.0e-5
    @test abs(
        FrankWolfe.lazified_conditional_gradient(
            f,
            grad!,
            lmo_prob,
            x0,
            max_iteration=1000,
            line_search=FrankWolfe.Shortstep(2.0),
            verbose=false,
        )[3] - 0.2,
    ) < 1.0e-5
    @test abs(
        FrankWolfe.lazified_conditional_gradient(
            f,
            grad!,
            lmo_prob,
            x0,
            max_iteration=1000,
            line_search=FrankWolfe.AdaptiveZerothOrder(),
            verbose=false,
        )[3] - 0.2,
    ) < 1.0e-5
end

@testset "Testing caching in Lazified Conditional Gradients" begin
    n = Int(1e5)
    L = 2
    k = 1000
    bound = 16 * L * 2 / (k + 2)

    f(x) = norm(x)^2
    function grad!(storage, x)
        @. storage = 2 * x
        return nothing
    end
    lmo_prob = FrankWolfe.ProbabilitySimplexOracle(1)
    x0 = FrankWolfe.compute_extreme_point(lmo_prob, zeros(n))

    x, v, primal, dual_gap, status, trajectory = FrankWolfe.lazified_conditional_gradient(
        f,
        grad!,
        lmo_prob,
        x0,
        max_iteration=k,
        line_search=FrankWolfe.Shortstep(2.0),
        verbose=false,
    )

    @test primal - 1 / n <= bound

    x, v, primal, dual_gap, status, trajectory = FrankWolfe.lazified_conditional_gradient(
        f,
        grad!,
        lmo_prob,
        x0,
        max_iteration=k,
        line_search=FrankWolfe.Shortstep(2.0),
        cache_size=100,
        verbose=false,
    )

    @test primal - 1 / n <= bound

    x, v, primal, dual_gap, status, trajectory = FrankWolfe.lazified_conditional_gradient(
        f,
        grad!,
        lmo_prob,
        x0,
        max_iteration=k,
        line_search=FrankWolfe.Shortstep(2.0),
        cache_size=100,
        greedy_lazy=true,
        verbose=false,
    )

    @test primal - 1 / n <= bound
end

@testset "Stochastic FW linear regression" begin
    function simple_reg_loss(θ, data_point)
        (xi, yi) = data_point
        (a, b) = (θ[1:(end-1)], θ[end])
        pred = a ⋅ xi + b
        return (pred - yi)^2 / 2
    end

    function ∇simple_reg_loss(storage, θ, data_point)
        (xi, yi) = data_point
        (a, b) = (θ[1:(end-1)], θ[end])
        pred = a ⋅ xi + b
        storage[1:(end-1)] .+= xi * (pred - yi)
        storage[end] += pred - yi
        return storage
    end

    xs = [10 * randn(rng, 5) for i in 1:20000]
    params = rand(rng, 6) .- 1 # start params in (-1,0)
    bias = 2π
    params_perfect = [1:5; bias]

    params = rand(rng, 6) .- 1 # start params in (-1,0)

    data_perfect = [(x, x ⋅ (1:5) + bias) for x in xs]
    f_stoch = FrankWolfe.StochasticObjective(
        simple_reg_loss,
        ∇simple_reg_loss,
        data_perfect,
        similar(params),
    )
    lmo = FrankWolfe.LpNormLMO{2}(1.1 * norm(params_perfect))

    θ, _, _, _, _ = FrankWolfe.stochastic_frank_wolfe(
        f_stoch,
        lmo,
        copy(params),
        momentum=0.95,
        verbose=false,
        line_search=FrankWolfe.Nonconvex(),
        max_iteration=100_000,
        batch_size=length(f_stoch.xs) ÷ 100,
        trajectory=false,
    )
    @test norm(θ - params_perfect) ≤ 0.05 * length(θ)

    # SFW with incrementing batch size
    batch_iterator =
        FrankWolfe.IncrementBatchIterator(length(f_stoch.xs) ÷ 1000, length(f_stoch.xs) ÷ 10, 2)
    θ, _, _, _, _ = FrankWolfe.stochastic_frank_wolfe(
        f_stoch,
        lmo,
        copy(params),
        momentum=0.95,
        verbose=false,
        line_search=FrankWolfe.Nonconvex(),
        max_iteration=5000,
        batch_iterator=batch_iterator,
        trajectory=false,
    )
    @test batch_iterator.maxreached
    # SFW damped momentum
    momentum_iterator = FrankWolfe.ExpMomentumIterator()
    θ, _, _, _, _ = FrankWolfe.stochastic_frank_wolfe(
        f_stoch,
        lmo,
        copy(params),
        verbose=false,
        line_search=FrankWolfe.Nonconvex(),
        max_iteration=5000,
        batch_size=1,
        trajectory=false,
        momentum_iterator=momentum_iterator,
    )
    θ, _, _, _, _ = FrankWolfe.stochastic_frank_wolfe(
        f_stoch,
        lmo,
        copy(params),
        line_search=FrankWolfe.Nonconvex(),
        max_iteration=5000,
        batch_size=1,
        verbose=false,
        trajectory=false,
        momentum_iterator=nothing,
    )
end

@testset "Away-step FW" begin
    n = 50
    lmo_prob = FrankWolfe.ProbabilitySimplexOracle(1.0)
    x0 = FrankWolfe.compute_extreme_point(lmo_prob, rand(rng, n))
    f(x) = norm(x)^2
    function grad!(storage, x)
        @. storage = 2x
    end
    k = 1000
    active_set = ActiveSet([(1.0, x0)])

    # compute reference from vanilla FW
    xref, _ = FrankWolfe.frank_wolfe(
        f,
        grad!,
        lmo_prob,
        x0,
        max_iteration=k,
        line_search=FrankWolfe.Backtracking(),
        verbose=false,
        memory_mode=FrankWolfe.OutplaceEmphasis(),
    )

    x, v, primal, dual_gap, status, trajectory = FrankWolfe.away_frank_wolfe(
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

    @test x !== nothing
    @test xref ≈ x atol = (1e-3 / length(x))

    xs, v, primal, dual_gap, status, trajectory = FrankWolfe.away_frank_wolfe(
        f,
        grad!,
        lmo_prob,
        active_set,
        max_iteration=k,
        line_search=FrankWolfe.Backtracking(),
        print_iter=k / 10,
        verbose=false,
        memory_mode=FrankWolfe.OutplaceEmphasis(),
    )

    @test xs !== nothing
    @test xref ≈ xs atol = (1e-3 / length(x))

    x, v, primal, dual_gap, status, trajectory = FrankWolfe.away_frank_wolfe(
        f,
        grad!,
        lmo_prob,
        x0,
        max_iteration=k,
        away_steps=false,
        line_search=FrankWolfe.Backtracking(),
        print_iter=k / 10,
        verbose=false,
        memory_mode=FrankWolfe.OutplaceEmphasis(),
    )

    @test x !== nothing
    @test xref ≈ x atol = (1e-3 / length(x))

    xs, v, primal, dual_gap, status, trajectory = FrankWolfe.away_frank_wolfe(
        f,
        grad!,
        lmo_prob,
        active_set,
        max_iteration=k,
        line_search=FrankWolfe.Backtracking(),
        print_iter=k / 10,
        verbose=false,
        memory_mode=FrankWolfe.OutplaceEmphasis(),
    )

    @test xs !== nothing
    @test xref ≈ xs atol = (1e-3 / length(x))

    x, v, primal, dual_gap, status, trajectory = FrankWolfe.away_frank_wolfe(
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

    @test x !== nothing
    @test xref ≈ x atol = (1e-3 / length(x))

    x, v, primal, dual_gap, status, trajectory = FrankWolfe.away_frank_wolfe(
        f,
        grad!,
        lmo_prob,
        x0,
        max_iteration=k,
        away_steps=false,
        line_search=FrankWolfe.Backtracking(),
        print_iter=k / 10,
        verbose=false,
        memory_mode=FrankWolfe.InplaceEmphasis(),
    )

    @test x !== nothing
    @test xref ≈ x atol = (1e-3 / length(x))

    xs, v, primal, dual_gap, status, trajectory = FrankWolfe.away_frank_wolfe(
        f,
        grad!,
        lmo_prob,
        active_set,
        max_iteration=k,
        line_search=FrankWolfe.Backtracking(),
        print_iter=k / 10,
        verbose=false,
        memory_mode=FrankWolfe.InplaceEmphasis(),
    )

    @test xs !== nothing
    @test xref ≈ xs atol = (1e-3 / length(x))

    empty!(active_set)
    @test_throws ArgumentError("Empty active set") FrankWolfe.away_frank_wolfe(
        f,
        grad!,
        lmo_prob,
        active_set,
        max_iteration=k,
        line_search=FrankWolfe.Backtracking(),
        print_iter=k / 10,
        verbose=false,
        memory_mode=FrankWolfe.OutplaceEmphasis(),
    )
end

@testset "Testing Blended Conditional Gradient" begin
    n = 50
    lmo_prob = FrankWolfe.ProbabilitySimplexOracle(1.0)
    x0 = FrankWolfe.compute_extreme_point(lmo_prob, randn(rng, n))
    f(x) = norm(x)^2
    function grad!(storage, x)
        @. storage = 2x
    end
    k = 1000

    # compute reference from vanilla FW
    xref, _ = FrankWolfe.frank_wolfe(
        f,
        grad!,
        lmo_prob,
        x0,
        max_iteration=k,
        line_search=FrankWolfe.Backtracking(),
        verbose=false,
        memory_mode=FrankWolfe.OutplaceEmphasis(),
    )

    x, v, primal, dual_gap, status, trajectory, _ = FrankWolfe.blended_conditional_gradient(
        f,
        grad!,
        lmo_prob,
        x0;
        line_search=FrankWolfe.Backtracking(),
        epsilon=1e-9,
        max_iteration=k,
        print_iter=1,
        trajectory=false,
        verbose=false,
    )

    @test x !== nothing
    @test f(x) ≈ f(xref)

end
