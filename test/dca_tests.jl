using FrankWolfe
using Test
using LinearAlgebra
using Random
using StableRNGs

@testset "DCA Algorithm Tests" begin

    # ==============================================================================
    # Test Setup and Helper Functions
    # ==============================================================================

    """
    Create a simple quadratic DC problem for testing:
    f(x) = 0.5 * ||x - a||²  (convex)
    g(x) = 0.5 * ||x - b||²  (convex)
    φ(x) = f(x) - g(x) = 0.5 * ||x - a||² - 0.5 * ||x - b||²

    This simplifies to φ(x) = (b - a)ᵀx + 0.5(||a||² - ||b||²)
    Which is linear, so the optimum is at a vertex of the feasible region.
    """
    function create_simple_dc_problem(n::Int, rng::StableRNG)
        a = randn(rng, n)
        b = randn(rng, n)

        f(x) = 0.5 * dot(x - a, x - a)
        function grad_f!(storage, x)
            storage .= x .- a
            return nothing
        end

        g(x) = 0.5 * dot(x - b, x - b)
        function grad_g!(storage, x)
            storage .= x .- b
            return nothing
        end

        # Combined objective for verification
        phi(x) = f(x) - g(x)

        return f, grad_f!, g, grad_g!, phi, a, b
    end

    """
    Create a more complex quadratic DC problem with matrices:
    f(x) = 0.5 * xᵀAx + aᵀx + c
    g(x) = 0.5 * xᵀBx + bᵀx + d
    """
    function create_quadratic_dc_problem(n::Int, rng::StableRNG)
        # Create positive definite matrices
        A_raw = randn(rng, n, n)
        A = A_raw' * A_raw + 0.1 * I

        B_raw = randn(rng, n, n)
        B = B_raw' * B_raw + 0.1 * I

        a = randn(rng, n)
        b = randn(rng, n)
        c = randn(rng)
        d = randn(rng)

        f(x) = 0.5 * dot(x, A, x) + dot(a, x) + c
        function grad_f!(storage, x)
            mul!(storage, A, x)
            storage .+= a
            return nothing
        end

        g(x) = 0.5 * dot(x, B, x) + dot(b, x) + d
        function grad_g!(storage, x)
            mul!(storage, B, x)
            storage .+= b
            return nothing
        end

        phi(x) = f(x) - g(x)

        return f, grad_f!, g, grad_g!, phi
    end

    # ==============================================================================
    # Basic Functionality Tests
    # ==============================================================================

    @testset "Basic DCA Functionality" begin
        Random.seed!(StableRNG(42), 42)
        n = 5
        rng = StableRNG(42)

        # Create simple DC problem
        f, grad_f!, g, grad_g!, phi, a, b = create_simple_dc_problem(n, rng)

        # Test on probability simplex
        lmo = FrankWolfe.ProbabilitySimplexOracle(1.0)
        x0 = FrankWolfe.compute_extreme_point(lmo, randn(rng, n))

        # Run DCA algorithm
        result = FrankWolfe.dca_fw(
            f,
            grad_f!,
            g,
            grad_g!,
            lmo,
            x0,
            max_iteration=50,
            max_inner_iteration=1000,
            epsilon=1e-6,
            verbose=false,
        )

        @test haskey(result, :x)
        @test haskey(result, :primal)
        @test haskey(result, :dca_gap)
        @test haskey(result, :iterations)
        @test haskey(result, :traj_data)

        # Check feasibility
        @test abs(sum(result.x) - 1.0) < 1e-10
        @test all(result.x .>= -1e-10)

        # Check convergence
        @test result.dca_gap < 1e-5
        @test result.iterations > 0

        # Verify primal value consistency
        @test abs(result.primal - phi(result.x)) < 1e-10
    end

    @testset "DCA with Different LMOs" begin
        Random.seed!(StableRNG(123), 123)
        n = 4
        rng = StableRNG(123)

        f, grad_f!, g, grad_g!, phi = create_quadratic_dc_problem(n, rng)

        # Test with different LMOs
        lmos = [
            FrankWolfe.ProbabilitySimplexOracle(1.0),
            FrankWolfe.ScaledBoundL1NormBall(-ones(n), ones(n)),
            FrankWolfe.KSparseLMO(2, 1.0),
        ]

        for (i, lmo) in enumerate(lmos)
            x0 = FrankWolfe.compute_extreme_point(lmo, randn(rng, n))

            result = FrankWolfe.dca_fw(
                f,
                grad_f!,
                g,
                grad_g!,
                lmo,
                x0,
                max_iteration=30,
                max_inner_iteration=500,
                epsilon=1e-5,
                verbose=false,
            )

            @test result.dca_gap < 1e-4
            @test result.iterations > 0
            @test abs(result.primal - phi(result.x)) < 1e-10
        end
    end

    # ==============================================================================
    # Algorithm Variants Tests
    # ==============================================================================

    @testset "DCA Algorithm Variants" begin
        Random.seed!(StableRNG(456), 456)
        n = 6
        rng = StableRNG(456)

        f, grad_f!, g, grad_g!, phi = create_quadratic_dc_problem(n, rng)
        lmo = FrankWolfe.ProbabilitySimplexOracle(1.0)
        x0 = FrankWolfe.compute_extreme_point(lmo, randn(rng, n))

        # Test standard Frank-Wolfe subsolver
        result_fw = FrankWolfe.dca_fw(
            f,
            grad_f!,
            g,
            grad_g!,
            lmo,
            copy(x0),
            max_iteration=20,
            use_corrective_fw=false,
            epsilon=1e-5,
            verbose=false,
        )

        # Test BPCG subsolver
        result_bpcg = FrankWolfe.dca_fw(
            f,
            grad_f!,
            g,
            grad_g!,
            lmo,
            copy(x0),
            max_iteration=20,
            use_corrective_fw=true,
            epsilon=1e-5,
            verbose=false,
        )

        # Test with early stopping
        result_early = FrankWolfe.dca_fw(
            f,
            grad_f!,
            g,
            grad_g!,
            lmo,
            copy(x0),
            max_iteration=20,
            use_dca_early_stopping=true,
            epsilon=1e-5,
            verbose=false,
        )

        # Test boosted variant
        result_boosted = FrankWolfe.dca_fw(
            f,
            grad_f!,
            g,
            grad_g!,
            lmo,
            copy(x0),
            max_iteration=20,
            boosted=true,
            epsilon=1e-5,
            verbose=false,
        )

        # All variants should converge
        @test result_fw.dca_gap < 1e-4
        @test result_bpcg.dca_gap < 1e-4
        @test result_warm.dca_gap < 1e-4
        @test result_early.dca_gap < 1e-4
        @test result_boosted.dca_gap < 1e-4

        # All should be feasible and consistent
        for result in [result_fw, result_bpcg, result_warm, result_early, result_boosted]
            @test abs(sum(result.x) - 1.0) < 1e-10
            @test all(result.x .>= -1e-10)
            @test abs(result.primal - phi(result.x)) < 1e-10
        end
    end

    @testset "Line Search Methods" begin
        Random.seed!(StableRNG(789), 789)
        n = 5
        rng = StableRNG(789)

        f, grad_f!, g, grad_g!, phi = create_quadratic_dc_problem(n, rng)
        lmo = FrankWolfe.ProbabilitySimplexOracle(1.0)
        x0 = FrankWolfe.compute_extreme_point(lmo, randn(rng, n))

        line_searches = [FrankWolfe.Agnostic(), FrankWolfe.Secant(), FrankWolfe.Adaptive()]

        for ls in line_searches
            result = FrankWolfe.dca_fw(
                f,
                grad_f!,
                g,
                grad_g!,
                lmo,
                copy(x0),
                max_iteration=25,
                line_search=ls,
                epsilon=1e-5,
                verbose=false,
            )

            @test result.dca_gap < 1e-4
            @test abs(result.primal - phi(result.x)) < 1e-10
        end
    end

    # ==============================================================================
    # Convergence and Mathematical Properties Tests
    # ==============================================================================

    @testset "DCA Gap Convergence" begin
        Random.seed!(StableRNG(101), 101)
        n = 4
        rng = StableRNG(101)

        f, grad_f!, g, grad_g!, phi = create_quadratic_dc_problem(n, rng)
        lmo = FrankWolfe.ProbabilitySimplexOracle(1.0)
        x0 = FrankWolfe.compute_extreme_point(lmo, randn(rng, n))

        result = FrankWolfe.dca_fw(
            f,
            grad_f!,
            g,
            grad_g!,
            lmo,
            x0,
            max_iteration=100,
            trajectory=true,
            epsilon=1e-8,
            verbose=false,
        )

        # Check that DCA gap decreases over time (generally)
        if length(result.traj_data) > 5
            # Trajectory data format: (t, primal, dual, dual_gap, time)
            gaps = [state[4] for state in result.traj_data[end-4:end]]  # dual_gap is 4th element
            # Allow for some non-monotonicity but overall trend should be decreasing
            @test gaps[end] <= gaps[1] * 10  # Allow some slack for numerical issues
        end

        @test result.dca_gap < 1e-7
    end

    @testset "Linear DC Problem - Exact Solution" begin
        Random.seed!(StableRNG(202), 202)
        n = 5
        rng = StableRNG(202)

        # For linear DC problem, we can verify against exact solution
        f, grad_f!, g, grad_g!, phi, a, b = create_simple_dc_problem(n, rng)
        lmo = FrankWolfe.ProbabilitySimplexOracle(1.0)

        # The exact optimal solution for linear φ(x) = (b-a)ᵀx + const
        # is the vertex that minimizes <b-a, x>, which is argmin <b-a, x>
        optimal_direction = b - a
        x_exact = FrankWolfe.compute_extreme_point(lmo, optimal_direction)
        exact_value = phi(x_exact)

        # Run DCA
        x0 = FrankWolfe.compute_extreme_point(lmo, randn(rng, n))
        result = FrankWolfe.dca_fw(
            f,
            grad_f!,
            g,
            grad_g!,
            lmo,
            x0,
            max_iteration=50,
            epsilon=1e-8,
            verbose=false,
        )

        # Should converge to the exact optimum (within tolerance)
        @test abs(result.primal - exact_value) < 1e-6
        @test result.dca_gap < 1e-7
    end

    # ==============================================================================
    # Edge Cases and Error Handling Tests
    # ==============================================================================

    @testset "Edge Cases" begin
        Random.seed!(StableRNG(303), 303)
        n = 3
        rng = StableRNG(303)

        f, grad_f!, g, grad_g!, phi = create_quadratic_dc_problem(n, rng)
        lmo = FrankWolfe.ProbabilitySimplexOracle(1.0)
        x0 = FrankWolfe.compute_extreme_point(lmo, randn(rng, n))

        # Test with max_iteration = 1
        result =
            FrankWolfe.dca_fw(f, grad_f!, g, grad_g!, lmo, copy(x0), max_iteration=1, verbose=false)
        @test result.iterations <= 1

        # Test with very loose tolerance
        result = FrankWolfe.dca_fw(
            f,
            grad_f!,
            g,
            grad_g!,
            lmo,
            copy(x0),
            max_iteration=100,
            epsilon=1e-1,
            verbose=false,
        )
        @test result.dca_gap < 1e-1

        # Test with timeout
        result = FrankWolfe.dca_fw(
            f,
            grad_f!,
            g,
            grad_g!,
            lmo,
            copy(x0),
            max_iteration=1000,
            timeout=0.01,  # Very short timeout
            verbose=false,
        )
        # Should terminate quickly due to timeout or converge very fast
        @test result.iterations >= 1
    end

    @testset "Memory Modes" begin
        Random.seed!(StableRNG(404), 404)
        n = 4
        rng = StableRNG(404)

        f, grad_f!, g, grad_g!, phi = create_quadratic_dc_problem(n, rng)
        lmo = FrankWolfe.ProbabilitySimplexOracle(1.0)
        x0 = FrankWolfe.compute_extreme_point(lmo, randn(rng, n))

        # Test InplaceEmphasis
        result_inplace = FrankWolfe.dca_fw(
            f,
            grad_f!,
            g,
            grad_g!,
            lmo,
            copy(x0),
            max_iteration=20,
            memory_mode=FrankWolfe.InplaceEmphasis(),
            epsilon=1e-5,
            verbose=false,
        )

        # Test OutplaceEmphasis  
        result_outplace = FrankWolfe.dca_fw(
            f,
            grad_f!,
            g,
            grad_g!,
            lmo,
            copy(x0),
            max_iteration=20,
            memory_mode=FrankWolfe.OutplaceEmphasis(),
            epsilon=1e-5,
            verbose=false,
        )

        # Both should work and converge
        @test result_inplace.dca_gap < 1e-4
        @test result_outplace.dca_gap < 1e-4
        @test abs(result_inplace.primal - phi(result_inplace.x)) < 1e-10
        @test abs(result_outplace.primal - phi(result_outplace.x)) < 1e-10
    end

    # ==============================================================================
    # Callback and Trajectory Tests
    # ==============================================================================

    @testset "Callbacks and Trajectories" begin
        Random.seed!(StableRNG(505), 505)
        n = 4
        rng = StableRNG(505)

        f, grad_f!, g, grad_g!, phi = create_quadratic_dc_problem(n, rng)
        lmo = FrankWolfe.ProbabilitySimplexOracle(1.0)
        x0 = FrankWolfe.compute_extreme_point(lmo, randn(rng, n))

        # Test trajectory collection
        result = FrankWolfe.dca_fw(
            f,
            grad_f!,
            g,
            grad_g!,
            lmo,
            x0,
            max_iteration=10,
            trajectory=true,
            verbose=false,
        )

        @test length(result.traj_data) > 0
        @test length(result.traj_data) <= result.iterations + 1  # +1 for final state

        # Check trajectory data structure (tuples with format (t, primal, dual, dual_gap, time))
        if length(result.traj_data) > 0
            state = result.traj_data[1]
            @test isa(state, Tuple)
            @test length(state) == 5
            @test isa(state[1], Integer)  # t (iteration)
            @test isa(state[2], Real)     # primal
            @test isa(state[3], Real)     # dual
            @test isa(state[4], Real)     # dual_gap
            @test isa(state[5], Real)     # time
        end

        # Test custom callback
        callback_calls = Ref(0)
        custom_callback = function (state)
            callback_calls[] += 1
            return true  # Continue optimization
        end

        result = FrankWolfe.dca_fw(
            f,
            grad_f!,
            g,
            grad_g!,
            lmo,
            copy(x0),
            max_iteration=5,
            callback=custom_callback,
            verbose=false,
        )

        @test callback_calls[] > 0
        @test callback_calls[] <= result.iterations + 1
    end

    # ==============================================================================
    # Performance and Scaling Tests
    # ==============================================================================

    @testset "Performance Tests" begin
        Random.seed!(StableRNG(606), 606)

        # Test different problem sizes
        for n in [5, 10, 20]
            rng = StableRNG(606 + n)
            f, grad_f!, g, grad_g!, phi = create_quadratic_dc_problem(n, rng)
            lmo = FrankWolfe.ProbabilitySimplexOracle(1.0)
            x0 = FrankWolfe.compute_extreme_point(lmo, randn(rng, n))

            # Run algorithm
            result = FrankWolfe.dca_fw(
                f,
                grad_f!,
                g,
                grad_g!,
                lmo,
                x0,
                max_iteration=15,
                epsilon=1e-4,
                verbose=false,
            )

            @test result.dca_gap < 1e-3
            @test abs(result.primal - phi(result.x)) < 1e-10
        end
    end

end  # End main testset
