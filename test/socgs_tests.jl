module Test_socgs_tests

using FrankWolfe
using Test
using LinearAlgebra
using Random
using StableRNGs
import HiGHS
import MathOptInterface as MOI

@testset "SOCGS Algorithm Tests" begin

    # ==============================================================================
    # Test Problems
    # ==============================================================================

    """
    Strongly convex quadratic f(x) = ½ xᵀQx + cᵀx.
    The quadratic model of SOCGS is exact for this problem.
    """
    function create_quadratic_problem(n::Int, rng::StableRNG)
        M = randn(rng, n, n)
        Q = M' * M / n + 0.1 * I
        c = randn(rng, n)

        f(x) = 0.5 * dot(x, Q, x) + dot(c, x)
        function grad!(storage, x)
            mul!(storage, Q, x)
            storage .+= c
            return nothing
        end
        function hessian!(H, x)
            H .= Q
            return nothing
        end
        return f, grad!, hessian!
    end

    """
    Quartic objective on the unit box from the SOCGS revisited paper:
    f(x) = 3x₁ + q₁/2 x₁² + ∑ᵢ₌₂ⁿ (qᵢ/2 (xᵢ - bᵢ)² + 5(xᵢ - bᵢ)⁴),
    whose minimizer over [0,1]ⁿ is x* = (0, b₂, …, bₙ) with a nonzero gradient in the first coordinate.
    """
    function create_quartic_box_problem(n::Int)
        q = collect(range(1.25, 3.0, length=n))
        b = [0.0; collect(range(0.25, 0.75, length=n - 1))]

        function f(x)
            value = 3 * x[1] + q[1] / 2 * x[1]^2
            for i in 2:n
                value += q[i] / 2 * (x[i] - b[i])^2 + 5 * (x[i] - b[i])^4
            end
            return value
        end
        function grad!(storage, x)
            storage[1] = 3 + q[1] * x[1]
            for i in 2:n
                storage[i] = q[i] * (x[i] - b[i]) + 20 * (x[i] - b[i])^3
            end
            return nothing
        end
        function hessian!(H, x)
            H .= 0
            H[1, 1] = q[1]
            for i in 2:n
                H[i, i] = q[i] + 60 * (x[i] - b[i])^2
            end
            return nothing
        end
        x_star = b
        return f, grad!, hessian!, x_star
    end

    # ==============================================================================
    # Convergence Tests
    # ==============================================================================

    @testset "Quadratic objective" begin
        n = 30
        rng = StableRNG(42)
        f, grad!, hessian! = create_quadratic_problem(n, rng)
        lmo = FrankWolfe.ProbabilitySimplexLMO(1.0)
        x0 = collect(FrankWolfe.compute_extreme_point(lmo, randn(rng, n)))

        reference = FrankWolfe.blended_pairwise_conditional_gradient(
            f,
            grad!,
            lmo,
            copy(x0),
            epsilon=1e-12,
            max_iteration=100000,
        )

        result = FrankWolfe.second_order_conditional_gradient_sliding(
            f,
            grad!,
            hessian!,
            lmo,
            copy(x0),
            epsilon=1e-8,
        )

        @test result.status == FrankWolfe.STATUS_OPTIMAL
        @test result.dual_gap ≤ 1e-8
        @test result.primal ≈ f(result.x)
        @test result.primal ≈ reference.primal atol = 1e-7
        @test sum(result.x) ≈ 1.0
        @test all(result.x .≥ -1e-12)
        @test result.x ≈ FrankWolfe.compute_active_set_iterate!(copy(result.active_set))
    end

    @testset "Non-quadratic objective with known minimizer" begin
        n = 20
        f, grad!, hessian!, x_star = create_quartic_box_problem(n)
        lmo = FrankWolfe.BoxLMO(zeros(n), ones(n))
        x0 = FrankWolfe.compute_extreme_point(lmo, ones(n))

        traj_data = []
        result = FrankWolfe.second_order_conditional_gradient_sliding(
            f,
            grad!,
            hessian!,
            lmo,
            copy(x0),
            epsilon=1e-10,
            trajectory=true,
            traj_data=traj_data,
        )

        @test result.status == FrankWolfe.STATUS_OPTIMAL
        @test result.dual_gap ≤ 1e-10
        @test norm(result.x - x_star) ≤ 1e-4
        @test result.primal ≈ f(x_star) atol = 1e-9

        # the objective value never increases
        primal_values = [state[2] for state in traj_data]
        @test all(diff(primal_values) .≤ 0)
        # SOCGS converges in few outer iterations on this problem
        @test traj_data[end][1] < 50
    end

    @testset "Corrective step variants" begin
        n = 20
        rng = StableRNG(123)
        f, grad!, hessian! = create_quadratic_problem(n, rng)
        lmo = FrankWolfe.ProbabilitySimplexLMO(1.0)
        x0 = collect(FrankWolfe.compute_extreme_point(lmo, randn(rng, n)))
        reference = FrankWolfe.blended_pairwise_conditional_gradient(
            f,
            grad!,
            lmo,
            copy(x0),
            epsilon=1e-12,
            max_iteration=100000,
        )

        steps = [
            FrankWolfe.AwayStep(),
            FrankWolfe.AwayStep(true),
            FrankWolfe.PairwiseStep(),
            FrankWolfe.BlendedPairwiseStep(true),
        ]
        @testset "$(typeof(step)) lazy=$(step.lazy)" for step in steps
            result = FrankWolfe.second_order_conditional_gradient_sliding(
                f,
                grad!,
                hessian!,
                lmo,
                copy(x0),
                epsilon=1e-8,
                inner_step=step,
                independent_step=step,
            )
            @test result.status == FrankWolfe.STATUS_OPTIMAL
            @test result.dual_gap ≤ 1e-8
            @test result.primal ≈ reference.primal atol = 1e-7
        end
    end

    @testset "Quadratic corrections" begin
        n = 20
        f, grad!, hessian!, x_star = create_quartic_box_problem(n)
        lmo = FrankWolfe.BoxLMO(zeros(n), ones(n))
        x0 = FrankWolfe.compute_extreme_point(lmo, ones(n))
        lp_optimizer() =
            MOI.instantiate(MOI.OptimizerWithAttributes(HiGHS.Optimizer, MOI.Silent() => true))

        builders = [
            "QC-LS" =>
                (A, b) -> FrankWolfe.ScheduledStep(
                    FrankWolfe.BlendedPairwiseStep(true),
                    FrankWolfe.QuadraticLSCorrection(A, b, false),
                ),
            "QC-LS-MNP" =>
                (A, b) -> FrankWolfe.ScheduledStep(
                    FrankWolfe.BlendedPairwiseStep(true),
                    FrankWolfe.QuadraticLSCorrection(A, b, true),
                ),
            "QC-LP" =>
                (A, b) -> FrankWolfe.ScheduledStep(
                    FrankWolfe.BlendedPairwiseStep(true),
                    FrankWolfe.QuadraticLPCorrection(A, b, lp_optimizer(), false),
                ),
            "QC-LP-MNP" =>
                (A, b) -> FrankWolfe.ScheduledStep(
                    FrankWolfe.BlendedPairwiseStep(true),
                    FrankWolfe.QuadraticLPCorrection(A, b, lp_optimizer(), true),
                ),
        ]
        @testset "$name" for (name, builder) in builders
            result = FrankWolfe.second_order_conditional_gradient_sliding(
                f,
                grad!,
                hessian!,
                lmo,
                copy(x0),
                epsilon=1e-8,
                inner_step=builder,
            )
            @test result.status == FrankWolfe.STATUS_OPTIMAL
            @test result.dual_gap ≤ 1e-8
            @test norm(result.x - x_star) ≤ 1e-3
        end

        @testset "Builder receives the quadratic model" begin
            x = fill(0.5, n)
            H = zeros(n, n)
            hessian!(H, x)
            gradient = similar(x)
            grad!(gradient, x)
            built = Ref{Any}(nothing)
            builder = function (A, b)
                built[] = (copy(A), copy(b))
                return FrankWolfe.BlendedPairwiseStep()
            end
            step = FrankWolfe._socgs_inner_step(builder, H, x, gradient, similar(x))
            @test step isa FrankWolfe.BlendedPairwiseStep
            A, b = built[]
            # ∇m(w) = A w + b with ∇m(x) = ∇f(x)
            @test A * x + b ≈ gradient
            @test A == H
        end
    end

    @testset "Inexact Hessian" begin
        n = 20
        f, grad!, hessian!, x_star = create_quartic_box_problem(n)
        lmo = FrankWolfe.BoxLMO(zeros(n), ones(n))
        x0 = FrankWolfe.compute_extreme_point(lmo, ones(n))

        # Hessian evaluated once at the starting point and never updated
        hessian_calls = Ref(0)
        function frozen_hessian!(H, x)
            if hessian_calls[] == 0
                hessian!(H, x)
            end
            hessian_calls[] += 1
            return nothing
        end
        result = FrankWolfe.second_order_conditional_gradient_sliding(
            f,
            grad!,
            frozen_hessian!,
            lmo,
            copy(x0),
            epsilon=1e-7,
            max_iteration=500,
        )
        @test result.status == FrankWolfe.STATUS_OPTIMAL
        @test result.dual_gap ≤ 1e-7
        @test norm(result.x - x_star) ≤ 1e-3
        @test hessian_calls[] > 1
    end

    @testset "Warm start" begin
        n = 20
        f, grad!, hessian!, x_star = create_quartic_box_problem(n)
        lmo = FrankWolfe.BoxLMO(zeros(n), ones(n))
        x0 = FrankWolfe.compute_extreme_point(lmo, ones(n))
        result = FrankWolfe.second_order_conditional_gradient_sliding(
            f,
            grad!,
            hessian!,
            lmo,
            copy(x0),
            epsilon=1e-8,
            warm_start=true,
        )
        @test result.status == FrankWolfe.STATUS_OPTIMAL
        @test result.dual_gap ≤ 1e-8
        @test norm(result.x - x_star) ≤ 1e-3
    end

    @testset "Matrix-shaped iterates" begin
        n = 4
        rng = StableRNG(7)
        Y = randn(rng, n, n)
        f(X) = 0.5 * norm(X - Y)^2 + 0.25 * sum((X - Y) .^ 4)
        function grad!(storage, X)
            storage .= (X .- Y) .+ (X .- Y) .^ 3
            return nothing
        end
        function hessian!(H, X)
            H .= 0
            H[diagind(H)] .= 1 .+ 3 .* vec(X .- Y) .^ 2
            return nothing
        end
        lmo = FrankWolfe.BirkhoffPolytopeLMO()
        x0 = collect(FrankWolfe.compute_extreme_point(lmo, randn(rng, n, n)))
        reference = FrankWolfe.blended_pairwise_conditional_gradient(
            f,
            grad!,
            lmo,
            copy(x0),
            epsilon=1e-12,
            max_iteration=100000,
        )
        result = FrankWolfe.second_order_conditional_gradient_sliding(
            f,
            grad!,
            hessian!,
            lmo,
            copy(x0),
            epsilon=1e-8,
        )
        @test result.status == FrankWolfe.STATUS_OPTIMAL
        @test result.x isa Matrix
        @test result.primal ≈ reference.primal atol = 1e-7
        @test sum(result.x, dims=1) ≈ ones(1, n)
        @test sum(result.x, dims=2) ≈ ones(n, 1)
    end

    # ==============================================================================
    # Inner Stopping Criterion
    # ==============================================================================

    @testset "Inner stopping criterion" begin
        n = 20
        f, grad!, hessian!, _ = create_quartic_box_problem(n)
        lmo = FrankWolfe.BoxLMO(zeros(n), ones(n))
        x_k = fill(0.6, n)
        tau = 1.0

        H = zeros(n, n)
        hessian!(H, x_k)
        gradient = similar(x_k)
        grad!(gradient, x_k)

        @testset "$(typeof(step)) lazy=$(step.lazy)" for step in [
            FrankWolfe.BlendedPairwiseStep(false),
            FrankWolfe.BlendedPairwiseStep(true),
        ]
            tracking_lmo = FrankWolfe.TrackingLMO(lmo)
            inner_gradient = similar(x_k)
            inner_lmo = FrankWolfe.SOCGSInnerLMO(tracking_lmo, inner_gradient)
            model_value, model_gradient! = FrankWolfe._socgs_model(
                x_k,
                f(x_k),
                gradient,
                H,
                similar(x_k),
                similar(x_k),
                inner_lmo,
            )
            # the model and its gradient coincide with f at the anchor
            @test model_value(x_k) ≈ f(x_k)
            storage = similar(x_k)
            model_gradient!(storage, x_k)
            @test storage ≈ gradient

            # x_k is not a vertex, start from the convex combination of two vertices
            v1 = FrankWolfe.compute_extreme_point(lmo, -ones(n))
            v2 = FrankWolfe.compute_extreme_point(lmo, ones(n))
            active_set = FrankWolfe.ActiveSet([(0.6, v1), (0.4, v2)])
            @test FrankWolfe.get_active_set_iterate(active_set) ≈ x_k
            criterion = FrankWolfe.SOCGSInnerStoppingCriterion(
                x_k,
                tau,
                inner_lmo,
                similar(x_k),
                active_set,
            )

            FrankWolfe.corrective_frank_wolfe(
                model_value,
                model_gradient!,
                inner_lmo,
                step,
                active_set,
                epsilon=0.0,
                max_iteration=10000,
                gradient=inner_gradient,
                callback=criterion,
                recompute_last_vertex=false,
            )
            @test criterion.satisfied
            @test criterion.active_set !== nothing

            # verify the criterion at the stored point with a separate LMO call
            z = FrankWolfe.get_active_set_iterate(criterion.active_set)
            model_gradient!(storage, z)
            v = FrankWolfe.compute_extreme_point(lmo, storage)
            gap = dot(storage, z - v)
            @test gap ≈ criterion.gap atol = 1e-12
            @test gap ≤ tau * norm(z - x_k)^4 + 1e-12
            @test model_value(z) ≤ model_value(x_k)

            # the criterion only reuses the LMO calls of the inner solver
            if !step.lazy
                # one LMO call at the start and one per subsequent iteration
                @test tracking_lmo.counter == criterion.iterations
            else
                @test tracking_lmo.counter ≤ criterion.iterations
            end
        end
    end

    @testset "Exact line search on the model" begin
        n = 10
        rng = StableRNG(3)
        M = randn(rng, n, n)
        H = M' * M + I
        x = randn(rng, n)
        gradient = randn(rng, n)
        d = randn(rng, n)
        workspace = FrankWolfe.SOCGSExactLineSearchWorkspace(H, similar(x))
        model(γ) = -γ * dot(gradient, d) + γ^2 / 2 * dot(d, H, d)
        gamma_exact = dot(gradient, d) / dot(d, H, d)
        for gamma_max in (abs(gamma_exact) / 2, 2 * abs(gamma_exact))
            gamma = FrankWolfe.perform_line_search(
                FrankWolfe.SOCGSExactLineSearch(),
                1,
                nothing,
                nothing,
                gradient,
                x,
                d,
                gamma_max,
                workspace,
                FrankWolfe.InplaceEmphasis(),
            )
            @test 0 ≤ gamma ≤ gamma_max
            # the step minimizes the model over [0, gamma_max]
            for γ in range(0, gamma_max, length=11)
                @test model(gamma) ≤ model(γ) + 1e-12
            end
        end
        @test_throws ArgumentError FrankWolfe.build_linesearch_workspace(
            FrankWolfe.SOCGSExactLineSearch(),
            x,
            gradient,
        )
    end

    # ==============================================================================
    # Options and Termination Status
    # ==============================================================================

    @testset "Options and termination status" begin
        n = 20
        f, grad!, hessian!, _ = create_quartic_box_problem(n)
        lmo = FrankWolfe.BoxLMO(zeros(n), ones(n))
        x0 = FrankWolfe.compute_extreme_point(lmo, ones(n))

        result = FrankWolfe.second_order_conditional_gradient_sliding(
            f,
            grad!,
            hessian!,
            lmo,
            copy(x0),
            epsilon=1e-14,
            max_iteration=1,
        )
        @test result.status == FrankWolfe.STATUS_MAXITER
        @test result.primal < f(x0)

        iterations = Ref(0)
        step_types = FrankWolfe.StepType[]
        callback = function (state, active_set)
            iterations[] = state.t
            push!(step_types, state.step_type)
            @test active_set isa FrankWolfe.AbstractActiveSet
            return state.t < 2
        end
        result = FrankWolfe.second_order_conditional_gradient_sliding(
            f,
            grad!,
            hessian!,
            lmo,
            copy(x0),
            epsilon=1e-14,
            callback=callback,
        )
        @test result.status == FrankWolfe.STATUS_INTERRUPTED
        @test iterations[] == 2
        @test step_types[1] in
              (FrankWolfe.ST_SOCGS_PVM, FrankWolfe.ST_SOCGS_CG, FrankWolfe.ST_SOCGS_KEEP)
        @test step_types[end] == FrankWolfe.ST_LAST

        result = FrankWolfe.second_order_conditional_gradient_sliding(
            f,
            grad!,
            hessian!,
            lmo,
            copy(x0),
            timeout=0.0,
        )
        @test result.status == FrankWolfe.STATUS_TIMEOUT
        @test result.x == x0

        result = FrankWolfe.second_order_conditional_gradient_sliding(
            f,
            grad!,
            hessian!,
            lmo,
            copy(x0),
            epsilon=1e-6,
            verbose=true,
            inner_verbose=true,
            inner_print_iter=10,
        )
        @test result.status == FrankWolfe.STATUS_OPTIMAL

        # a positive inner tolerance still converges
        result = FrankWolfe.second_order_conditional_gradient_sliding(
            f,
            grad!,
            hessian!,
            lmo,
            copy(x0),
            epsilon=1e-8,
            inner_epsilon=1e-9,
        )
        @test result.status == FrankWolfe.STATUS_OPTIMAL

        # generic line searches for the subproblems
        result = FrankWolfe.second_order_conditional_gradient_sliding(
            f,
            grad!,
            hessian!,
            lmo,
            copy(x0),
            epsilon=1e-7,
            inner_line_search=FrankWolfe.Secant(),
        )
        @test result.status == FrankWolfe.STATUS_OPTIMAL

        @test_throws ArgumentError FrankWolfe.second_order_conditional_gradient_sliding(
            f,
            grad!,
            hessian!,
            lmo,
            copy(x0),
            tau=0.0,
        )
    end
end

end # module
