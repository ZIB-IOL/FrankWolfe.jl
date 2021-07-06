using FrankWolfe
using Test
using LinearAlgebra
using DoubleFloats

include("lmo.jl")
include("function_gradient.jl")
include("active_set.jl")
include("utils.jl")

@testset "Testing vanilla Frank-Wolfe with various step size and momentum strategies" begin
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
            max_iteration=1000,
            line_search=FrankWolfe.Agnostic(),
            verbose=true,
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
            verbose=true,
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
        line_search=FrankWolfe.Shortstep(),
        L=2,
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
            emphasis=FrankWolfe.memory,
        )[3] - 0.2,
    ) < 1.0e-3
    @test abs(
        FrankWolfe.frank_wolfe(
            f,
            grad!,
            lmo_prob,
            x0,
            max_iteration=1000,
            line_search=FrankWolfe.Adaptive(),
            L=100,
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
            line_search=FrankWolfe.Adaptive(),
            L=100,
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
            line_search=FrankWolfe.Adaptive(),
            L=100,
            verbose=false,
            momentum=0.9,
            emphasis=FrankWolfe.memory,
        )[3] - 0.2,
    ) < 1.0e-3
end

@testset "Gradient with momentum correctly updated" begin
    # fixing https://github.com/ZIB-IOL/FrankWolfe.jl/issues/47
    include("momentum_memory.jl")
end
@testset "Testing Lazified Conditional Gradients with various step size strategies" begin
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
            verbose=true,
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
            verbose=true,
        )[3] - 0.2,
    ) < 1.0e-5
    @test abs(
        FrankWolfe.lazified_conditional_gradient(
            f,
            grad!,
            lmo_prob,
            x0,
            max_iteration=1000,
            line_search=FrankWolfe.Shortstep(),
            L=2,
            verbose=true,
        )[3] - 0.2,
    ) < 1.0e-5
end

@testset "Testing Lazified Conditional Gradients with cache strategies" begin
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

    @time x, v, primal, dual_gap, trajectory = FrankWolfe.lazified_conditional_gradient(
        f,
        grad!,
        lmo_prob,
        x0,
        max_iteration=k,
        line_search=FrankWolfe.Shortstep(),
        L=2,
        verbose=true,
    )

    @test primal - 1 / n <= bound

    @time x, v, primal, dual_gap, trajectory = FrankWolfe.lazified_conditional_gradient(
        f,
        grad!,
        lmo_prob,
        x0,
        max_iteration=k,
        line_search=FrankWolfe.Shortstep(),
        L=2,
        cache_size=100,
        verbose=false,
    )

    @test primal - 1 / n <= bound

    @time x, v, primal, dual_gap, trajectory = FrankWolfe.lazified_conditional_gradient(
        f,
        grad!,
        lmo_prob,
        x0,
        max_iteration=k,
        line_search=FrankWolfe.Shortstep(),
        L=2,
        cache_size=100,
        greedy_lazy=true,
        verbose=false,
    )

    @test primal - 1 / n <= bound
end

@testset "Testing emphasis blas vs memory" begin
    n = Int(1e5)
    k = 100
    xpi = rand(n)
    total = sum(xpi)
    xp = xpi ./ total
    f(x) = norm(x - xp)^2
    function grad!(storage, x)
        @. storage = 2 * (x - xp)
        return nothing
    end
    @testset "Using sparse structure" begin
        lmo_prob = FrankWolfe.ProbabilitySimplexOracle(1.0)
        x0 = FrankWolfe.compute_extreme_point(lmo_prob, zeros(n))

        x, v, primal, dual_gap, trajectory = FrankWolfe.frank_wolfe(
            f,
            grad!,
            lmo_prob,
            x0,
            max_iteration=k,
            line_search=FrankWolfe.Backtracking(),
            print_iter=k / 10,
            verbose=false,
            emphasis=FrankWolfe.blas,
        )

        @test x !== nothing

        x, v, primal, dual_gap, trajectory = FrankWolfe.frank_wolfe(
            f,
            grad!,
            lmo_prob,
            x0,
            max_iteration=k,
            line_search=FrankWolfe.Backtracking(),
            print_iter=k / 10,
            verbose=false,
            emphasis=FrankWolfe.memory,
        )

        @test x !== nothing
    end
    @testset "Using dense structure" begin
        lmo_prob = FrankWolfe.L1ballDense{Float64}(1)
        x0 = FrankWolfe.compute_extreme_point(lmo_prob, zeros(n))

        x, _ = FrankWolfe.frank_wolfe(
            f,
            grad!,
            lmo_prob,
            copy(x0),
            max_iteration=k,
            line_search=FrankWolfe.Backtracking(),
            print_iter=k / 10,
            verbose=false,
            emphasis=FrankWolfe.blas,
        )

        @test x !== nothing

        x, _ = FrankWolfe.frank_wolfe(
            f,
            grad!,
            lmo_prob,
            copy(x0),
            max_iteration=k,
            line_search=FrankWolfe.Backtracking(),
            print_iter=k / 10,
            verbose=false,
            emphasis=FrankWolfe.memory,
        )

        @test x !== nothing

        line_search = FrankWolfe.MonotonousStepSize()
        x, _, primal_conv, _ = FrankWolfe.frank_wolfe(
            f,
            grad!,
            lmo_prob,
            copy(x0),
            max_iteration=k,
            line_search=line_search,
            print_iter=k / 10,
            verbose=false,
            emphasis=FrankWolfe.memory,
        )
        @test line_search.factor < 20

        line_search = FrankWolfe.MonotonousNonConvexStepSize()
        x, _, primal_nonconv, _ = FrankWolfe.frank_wolfe(
            f,
            grad!,
            lmo_prob,
            copy(x0),
            max_iteration=k,
            line_search=line_search,
            print_iter=k / 10,
            verbose=false,
            emphasis=FrankWolfe.memory,
        )
        @test line_search.factor < 20
    end
end
@testset "Testing rational variant" begin
    rhs = 1
    n = 40
    k = 1000

    xpi = rand(big(1):big(100), n)
    total = sum(xpi)
    xp = xpi .// total

    f(x) = norm(x - xp)^2
    function grad!(storage, x)
        @. storage = 2 * (x - xp)
    end

    lmo = FrankWolfe.ProbabilitySimplexOracle{Rational{BigInt}}(rhs)
    direction = rand(n)
    x0 = FrankWolfe.compute_extreme_point(lmo, direction)
    @test eltype(x0) == Rational{BigInt}

    @time x, v, primal, dual_gap, trajectory = FrankWolfe.frank_wolfe(
        f,
        grad!,
        lmo,
        x0,
        max_iteration=k,
        line_search=FrankWolfe.Agnostic(),
        print_iter=k / 10,
        emphasis=FrankWolfe.blas,
        verbose=false,
    )

    @test eltype(x0) == Rational{BigInt}

    @time x, v, primal, dual_gap, trajectory = FrankWolfe.frank_wolfe(
        f,
        grad!,
        lmo,
        x0,
        max_iteration=k,
        line_search=FrankWolfe.Agnostic(),
        print_iter=k / 10,
        emphasis=FrankWolfe.memory,
        verbose=true,
    )
    @test eltype(x0) == eltype(x) == Rational{BigInt}
    @test f(x) <= 1e-4

    # very slow computation, explodes quickly
    x0 = collect(FrankWolfe.compute_extreme_point(lmo, direction))
    @time x, v, primal, dual_gap, trajectory = FrankWolfe.frank_wolfe(
        f,
        grad!,
        lmo,
        x0,
        max_iteration=15,
        line_search=FrankWolfe.RationalShortstep(),
        L=2,
        print_iter=k / 100,
        emphasis=FrankWolfe.memory,
        verbose=true,
    )

    x0 = FrankWolfe.compute_extreme_point(lmo, direction)
    @time x, v, primal, dual_gap, trajectory = FrankWolfe.frank_wolfe(
        f,
        grad!,
        lmo,
        x0,
        max_iteration=15,
        line_search=FrankWolfe.RationalShortstep(),
        L=2,
        print_iter=k / 10,
        emphasis=FrankWolfe.memory,
        verbose=true,
    )
    @test eltype(x) == Rational{BigInt}
end
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
        println("\nTesting precision for type: ", T)
        lmo = FrankWolfe.ProbabilitySimplexOracle{T}(rhs)
        direction = rand(n)
        x0 = FrankWolfe.compute_extreme_point(lmo, direction)

        @time x, v, primal, dual_gap, trajectory = FrankWolfe.frank_wolfe(
            f,
            grad!,
            lmo,
            x0,
            max_iteration=k,
            line_search=FrankWolfe.Agnostic(),
            print_iter=k / 10,
            emphasis=FrankWolfe.blas,
            verbose=true,
        )

        @test eltype(x0) == T
        @test primal - 1 / n <= bound

        @time x, v, primal, dual_gap, trajectory = FrankWolfe.frank_wolfe(
            f,
            grad!,
            lmo,
            x0,
            max_iteration=k,
            line_search=FrankWolfe.Agnostic(),
            print_iter=k / 10,
            emphasis=FrankWolfe.memory,
            verbose=true,
        )

        @test eltype(x0) == T
        @test primal - 1 // n <= bound

        @time x, v, primal, dual_gap, trajectory = FrankWolfe.away_frank_wolfe(
            f,
            grad!,
            lmo,
            x0,
            max_iteration=k,
            line_search=FrankWolfe.Adaptive(),
            print_iter=k / 10,
            emphasis=FrankWolfe.memory,
            verbose=true,
        )

        @test eltype(x0) == T
        @test primal - 1 // n <= bound

        @time x, v, primal, dual_gap, trajectory = FrankWolfe.blended_conditional_gradient(
            f,
            grad!,
            lmo,
            x0,
            max_iteration=k,
            line_search=FrankWolfe.Adaptive(),
            print_iter=k / 10,
            emphasis=FrankWolfe.memory,
            verbose=true,
        )

        @test eltype(x0) == T
        @test primal - 1 // n <= bound



    end
end

@testset "Stochastic FW linear regression" begin
    function simple_reg_loss(θ, data_point)
        (xi, yi) = data_point
        (a, b) = (θ[1:end-1], θ[end])
        pred = a ⋅ xi + b
        return (pred - yi)^2 / 2
    end

    function ∇simple_reg_loss(θ, data_point)
        (xi, yi) = data_point
        (a, b) = (θ[1:end-1], θ[end])
        pred = a ⋅ xi + b
        grad_a = xi * (pred - yi)
        grad = push!(grad_a, pred - yi)
        return grad
    end

    xs = [10 * randn(5) for i in 1:20000]
    params = rand(6) .- 1 # start params in (-1,0)
    bias = 2π
    params_perfect = [1:5; bias]

    params = rand(6) .- 1 # start params in (-1,0)

    data_perfect = [(x, x ⋅ (1:5) + bias) for x in xs]
    f_stoch = FrankWolfe.StochasticObjective(simple_reg_loss, ∇simple_reg_loss, data_perfect)
    lmo = FrankWolfe.LpNormLMO{2}(1.1 * norm(params_perfect))

    θ, _, _, _, _ = FrankWolfe.stochastic_frank_wolfe(
        f_stoch,
        lmo,
        params,
        momentum=0.95,
        verbose=false,
        line_search=FrankWolfe.Nonconvex(),
        max_iteration=50_000,
        batch_size=length(f_stoch.xs) ÷ 100,
        trajectory=false,
    )
    @test norm(θ - params_perfect) ≤ 0.025 * length(θ)
end

@testset "Away-step FW" begin
    n = 50
    lmo_prob = FrankWolfe.ProbabilitySimplexOracle(1.0)
    x0 = FrankWolfe.compute_extreme_point(lmo_prob, rand(n))
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
        emphasis=FrankWolfe.blas,
    )

    x, v, primal, dual_gap, trajectory = FrankWolfe.away_frank_wolfe(
        f,
        grad!,
        lmo_prob,
        x0,
        max_iteration=k,
        line_search=FrankWolfe.Backtracking(),
        print_iter=k / 10,
        verbose=true,
        emphasis=FrankWolfe.blas,
    )

    @test x !== nothing
    @test xref ≈ x atol = (1e-3 / length(x))

    x, v, primal, dual_gap, trajectory = FrankWolfe.away_frank_wolfe(
        f,
        grad!,
        lmo_prob,
        x0,
        max_iteration=k,
        line_search=FrankWolfe.Backtracking(),
        print_iter=k / 10,
        verbose=true,
        emphasis=FrankWolfe.memory,
    )
    @test x !== nothing
    @test xref ≈ x atol = (1e-3 / length(x))
end

@testset "Blended conditional gradient" begin
    n = 50
    lmo_prob = FrankWolfe.ProbabilitySimplexOracle(1.0)
    x0 = FrankWolfe.compute_extreme_point(lmo_prob, randn(n))
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
        verbose=true,
        emphasis=FrankWolfe.blas,
    )

    x, v, primal, dual_gap, trajectory = FrankWolfe.blended_conditional_gradient(
        f,
        grad!,
        lmo_prob,
        x0;
        line_search=FrankWolfe.Backtracking(),
        L=Inf,
        epsilon=1e-9,
        max_iteration=k,
        print_iter=1,
        trajectory=false,
        verbose=false,
        linesearch_tol=1e-10,
    )

    @test x !== nothing
    @test f(x) ≈ f(xref)

end


include("oddities.jl")


# in separate module for name space issues
module BCGDirectionError
using Test
@testset "BCG direction accuracy" begin
    include("bcg_direction_error.jl")
end
end

module RationalTest
using Test
@testset "Rational test and shortstep" begin
    include("rational_test.jl")
end
end

if get(ENV, "FW_TEST", nothing) == "full"
    @testset "Running examples" begin
        # TODO test smaller examples to be sure they are up to date
    end
end

