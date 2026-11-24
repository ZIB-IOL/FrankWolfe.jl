module Test_socgs_tests

using FrankWolfe
using Test
using LinearAlgebra
using Random
using StableRNGs

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
        x0 = FrankWolfe.compute_extreme_point(lmo, randn(rng, n))

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
        x0 = FrankWolfe.compute_extreme_point(lmo, randn(rng, n))
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

    @testset "Corrective step builders" begin
        n = 20
        f, grad!, hessian!, x_star = create_quartic_box_problem(n)
        lmo = FrankWolfe.BoxLMO(zeros(n), ones(n))
        x0 = FrankWolfe.compute_extreme_point(lmo, ones(n))

        builder_calls = Ref(0)
        builder = function (A, b)
            builder_calls[] += 1
            return FrankWolfe.BlendedPairwiseStep(true)
        end
        iterations = Ref(0)
        callback = function (state, active_set)
            iterations[] = state.t
            return true
        end
        result = FrankWolfe.second_order_conditional_gradient_sliding(
            f,
            grad!,
            hessian!,
            lmo,
            copy(x0),
            epsilon=1e-8,
            inner_step=builder,
            callback=callback,
        )
        @test result.status == FrankWolfe.STATUS_OPTIMAL
        @test result.dual_gap ≤ 1e-8
        @test norm(result.x - x_star) ≤ 1e-3
        # one corrective step per subproblem
        @test builder_calls[] == iterations[]

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

    @testset "Rounding errors in the objective value" begin
        n = 20
        f, grad!, hessian!, x_star = create_quartic_box_problem(n)
        lmo = FrankWolfe.BoxLMO(zeros(n), ones(n))
        x0 = FrankWolfe.compute_extreme_point(lmo, ones(n))
        # the shift hides the decrease of f close to the solution in rounding errors relative to f,
        # and the Hessian approximation gives linear convergence, so this regime spans several iterations
        function scaled_hessian!(H, x)
            hessian!(H, x)
            H .*= 2
            return nothing
        end

        # objective values of the outer iterates x_0, …, x_K, the last one is reported with ST_LAST
        primal_values = Float64[]
        callback = function (state, active_set)
            state.step_type != FrankWolfe.ST_POSTPROCESS && push!(primal_values, state.primal)
            return true
        end
        result = FrankWolfe.second_order_conditional_gradient_sliding(
            x -> f(x) + 1,
            grad!,
            scaled_hessian!,
            lmo,
            copy(x0),
            epsilon=1e-10,
            max_iteration=100,
            callback=callback,
        )
        @test result.status == FrankWolfe.STATUS_OPTIMAL
        @test result.dual_gap ≤ 1e-10
        @test norm(result.x - x_star) ≤ 1e-4
        # z_k or a_{k+1} are selected although their objective values are not smaller,
        # but only within the tolerance
        increases = diff(primal_values)
        @test any(≥(0), increases)
        @test all(increases .< 10 .* abs.(primal_values[1:(end-1)]) .* eps())
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
        x0 = FrankWolfe.compute_extreme_point(lmo, randn(rng, n, n))
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
        @test result.x isa AbstractMatrix
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

        steps = [FrankWolfe.BlendedPairwiseStep(false), FrankWolfe.BlendedPairwiseStep(true)]
        @testset "$(typeof(step)) lazy=$(step.lazy)" for step in steps
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
            criterion = FrankWolfe.SOCGSInnerStoppingCriterion(x_k, tau, inner_lmo, similar(x_k))

            result = FrankWolfe.corrective_frank_wolfe(
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
            @test result.status == FrankWolfe.STATUS_INTERRUPTED

            # verify the criterion at the returned point with a separate LMO call
            z = result.x
            model_gradient!(storage, z)
            v = FrankWolfe.compute_extreme_point(lmo, storage)
            gap = dot(storage, z - v)
            @test gap ≈ criterion.gap atol = 1e-12
            tolerance = tau * norm(z - x_k)^4
            @test gap ≤ tolerance + 1e-12
            @test FrankWolfe._socgs_inner_tolerance(criterion, (x=z, gradient=storage)) ≈ tolerance
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

    @testset "Reuse of the vertex at the start of a subproblem" begin
        n = 5
        lmo = FrankWolfe.BoxLMO(zeros(n), ones(n))
        start_gradient = collect(1.0:n)
        start_vertex = FrankWolfe.compute_extreme_point(lmo, start_gradient)

        tracking_lmo = FrankWolfe.TrackingLMO(lmo)
        buffer = copy(start_gradient)
        inner_lmo = FrankWolfe.SOCGSInnerLMO(tracking_lmo, buffer, start_vertex, start_gradient)
        # other directions are passed on and do not use up the vertex
        FrankWolfe.compute_extreme_point(inner_lmo, copy(start_gradient))
        @test tracking_lmo.counter == 1
        @test !inner_lmo.fresh
        @test FrankWolfe.compute_extreme_point(inner_lmo, buffer) === start_vertex
        @test tracking_lmo.counter == 1
        @test inner_lmo.fresh
        @test inner_lmo.gradient_dot_vertex == dot(buffer, start_vertex)
        # the vertex is only used once
        @test FrankWolfe.compute_extreme_point(inner_lmo, buffer) == start_vertex
        @test tracking_lmo.counter == 2

        # the start of the subproblem differs from x_k
        tracking_lmo = FrankWolfe.TrackingLMO(lmo)
        buffer = -start_gradient
        inner_lmo = FrankWolfe.SOCGSInnerLMO(tracking_lmo, buffer, start_vertex, start_gradient)
        @test FrankWolfe.compute_extreme_point(inner_lmo, buffer) ==
              FrankWolfe.compute_extreme_point(lmo, buffer)
        @test tracking_lmo.counter == 1
        @test inner_lmo.fresh
    end

    # ==============================================================================
    # Independent Sequence
    # ==============================================================================

    @testset "Independent sequence" begin
        n = 20
        f, grad!, hessian!, _ = create_quartic_box_problem(n)
        lmo = FrankWolfe.BoxLMO(zeros(n), ones(n))
        x0 = FrankWolfe.compute_extreme_point(lmo, ones(n))
        num_steps = 15

        line_searches = (FrankWolfe.Secant(), FrankWolfe.Agnostic())
        steps = [
            FrankWolfe.BlendedPairwiseStep(),
            FrankWolfe.BlendedPairwiseStep(true),
            FrankWolfe.AwayStep(),
            FrankWolfe.PairwiseStep(),
        ]
        @testset "Matches corrective_frank_wolfe with $(typeof(line_search)) $(typeof(step)) lazy=$(step.lazy)" for line_search in
                                                                                                                    line_searches,
            step in steps

            tracking_grad! = FrankWolfe.TrackingGradient(grad!)
            tracking_lmo = FrankWolfe.TrackingLMO(lmo)
            reference_active_set = FrankWolfe.ActiveSet([(1.0, copy(x0))])
            # like the independent sequence, the reference recomputes its iterate from the active set
            # before evaluating f and the gradient at the start of every iteration
            function reference_f(x)
                if x === FrankWolfe.get_active_set_iterate(reference_active_set)
                    FrankWolfe.compute_active_set_iterate!(reference_active_set)
                end
                return f(x)
            end
            reference = FrankWolfe.corrective_frank_wolfe(
                reference_f,
                tracking_grad!,
                tracking_lmo,
                step,
                reference_active_set,
                line_search=line_search,
                epsilon=1e-14,
                max_iteration=num_steps - 1,
                recompute_last_vertex=false,
            )
            reference_gradient_calls = tracking_grad!.counter
            reference_lmo_calls = tracking_lmo.counter

            tracking_grad!.counter = 0
            tracking_lmo.counter = 0
            active_set = FrankWolfe.ActiveSet([(1.0, copy(x0))])
            x = FrankWolfe.get_active_set_iterate(active_set)
            gradient = similar(x)
            tracking_grad!(gradient, x)
            v = FrankWolfe.compute_extreme_point(tracking_lmo, gradient)
            dual_gap = dot(gradient, x) - dot(gradient, v)
            phi_value = dual_gap
            primal = f(x)
            vertex_is_current = true
            workspace = FrankWolfe.build_linesearch_workspace(line_search, x, gradient)
            for t in 1:num_steps
                primal, v, dual_gap, phi_value, vertex_is_current =
                    FrankWolfe._socgs_independent_step!(
                        step,
                        f,
                        tracking_grad!,
                        gradient,
                        active_set,
                        primal,
                        v,
                        dual_gap,
                        phi_value,
                        vertex_is_current,
                        t,
                        tracking_lmo,
                        line_search,
                        workspace,
                        0.0,
                        1000,
                        FrankWolfe.InplaceEmphasis(),
                        1e-14,
                        similar(x),
                    )
            end

            # the iterates only differ by the final cleanup of corrective_frank_wolfe
            @test FrankWolfe.get_active_set_iterate(active_set) ≈ reference.x atol = 1e-6
            @test primal == f(FrankWolfe.get_active_set_iterate(active_set))
            # the iterate is recomputed from the active set
            @test FrankWolfe.get_active_set_iterate(active_set) ==
                  FrankWolfe.compute_active_set_iterate!(copy(active_set))
            # the independent sequence evaluates the gradient at a_{k+1} at the end of every step, while
            # corrective_frank_wolfe evaluates it at the start of the next iteration, which the reference does not run
            @test tracking_grad!.counter == reference_gradient_calls + 1
            @test tracking_lmo.counter == reference_lmo_calls
        end

        @testset "Oracle calls of the algorithm" begin
            tracking_grad! = FrankWolfe.TrackingGradient(grad!)
            tracking_lmo = FrankWolfe.TrackingLMO(lmo)
            step_types = FrankWolfe.StepType[]
            callback = function (state, active_set)
                push!(step_types, state.step_type)
                return true
            end
            # Agnostic does not call the gradient, and one inner iteration only calls the LMO at its start
            result = FrankWolfe.second_order_conditional_gradient_sliding(
                f,
                tracking_grad!,
                hessian!,
                tracking_lmo,
                copy(x0),
                epsilon=1e-14,
                max_iteration=num_steps,
                line_search=FrankWolfe.Agnostic(),
                inner_max_iteration=0,
                callback=callback,
            )
            @test result.status == FrankWolfe.STATUS_MAXITER
            @test step_types[(end-1):end] == [FrankWolfe.ST_LAST, FrankWolfe.ST_POSTPROCESS]
            resize!(step_types, length(step_types) - 2)
            num_pvm = count(==(FrankWolfe.ST_SOCGS_PVM), step_types)
            num_cg = count(==(FrankWolfe.ST_SOCGS_CG), step_types)
            @test num_pvm > 0
            @test num_cg > 0

            # one gradient at x_0 and at every a_{k+1}, and one for every subproblem solution that is selected
            @test tracking_grad!.counter == 1 + num_steps + num_pvm
            # the vertex at x_0 and after a selected a_{k+1} is reused for the next independent step
            num_cg_reused = count(==(FrankWolfe.ST_SOCGS_CG), step_types[1:(end-1)])
            lmo_calls_independent = num_steps - 1 - num_cg_reused
            # the subproblems start from x_k and reuse the vertex at x_k for that call
            @test tracking_lmo.counter == 1 + lmo_calls_independent + num_pvm + num_cg
        end

        @testset "Periodic refresh of the active set" begin
            kwargs = (
                epsilon=1e-14,
                max_iteration=num_steps,
                inner_max_iteration=0,
                line_search=FrankWolfe.Agnostic(),
            )
            result = FrankWolfe.second_order_conditional_gradient_sliding(
                f,
                grad!,
                hessian!,
                lmo,
                copy(x0);
                independent_renorm_interval=2,
                kwargs...,
            )
            @test result.primal ≈
                  FrankWolfe.second_order_conditional_gradient_sliding(
                f,
                grad!,
                hessian!,
                lmo,
                copy(x0);
                kwargs...,
            ).primal rtol = 1e-6
        end

        @testset "Iterates match their active sets" begin
            # x_k is recomputed from its active set, so a subproblem starting from x_k reproduces it exactly
            variants = [
                "subproblem solutions" => (FrankWolfe.ST_SOCGS_PVM, (epsilon=1e-10,)),
                "independent sequence" => (
                    FrankWolfe.ST_SOCGS_CG,
                    (
                        epsilon=1e-14,
                        max_iteration=num_steps,
                        inner_max_iteration=0,
                        line_search=FrankWolfe.Agnostic(),
                    ),
                ),
            ]
            @testset "$name" for (name, (selected, kwargs)) in variants
                step_types = FrankWolfe.StepType[]
                consistent = Bool[]
                callback = function (state, active_set)
                    push!(step_types, state.step_type)
                    push!(
                        consistent,
                        state.x == FrankWolfe.compute_active_set_iterate!(copy(active_set)),
                    )
                    return true
                end
                FrankWolfe.second_order_conditional_gradient_sliding(
                    f,
                    grad!,
                    hessian!,
                    lmo,
                    copy(x0);
                    callback=callback,
                    kwargs...,
                )
                @test all(consistent)
                @test selected in step_types
            end
        end

        @testset "Without the independent sequence" begin
            tracking_grad! = FrankWolfe.TrackingGradient(grad!)
            tracking_lmo = FrankWolfe.TrackingLMO(lmo)
            step_types = FrankWolfe.StepType[]
            callback = function (state, active_set)
                push!(step_types, state.step_type)
                return true
            end
            result = FrankWolfe.second_order_conditional_gradient_sliding(
                f,
                tracking_grad!,
                hessian!,
                tracking_lmo,
                copy(x0),
                epsilon=1e-14,
                max_iteration=num_steps,
                inner_max_iteration=0,
                independent_sequence=false,
                callback=callback,
            )
            @test step_types[(end-1):end] == [FrankWolfe.ST_LAST, FrankWolfe.ST_POSTPROCESS]
            resize!(step_types, length(step_types) - 2)
            @test FrankWolfe.ST_SOCGS_CG ∉ step_types
            num_pvm = count(==(FrankWolfe.ST_SOCGS_PVM), step_types)
            @test num_pvm > 0
            # one gradient and one LMO call at x_0 and at every selected subproblem solution
            @test tracking_grad!.counter == 1 + num_pvm
            @test tracking_lmo.counter == 1 + num_pvm

            result = FrankWolfe.second_order_conditional_gradient_sliding(
                f,
                grad!,
                hessian!,
                lmo,
                copy(x0),
                independent_sequence=false,
            )
            @test result.status == FrankWolfe.STATUS_OPTIMAL
            @test result.dual_gap ≤ 1e-7
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
        @test step_types[(end-1):end] == [FrankWolfe.ST_LAST, FrankWolfe.ST_POSTPROCESS]

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

        # without the inner tolerance, the inner stopping criterion alone still converges
        result = FrankWolfe.second_order_conditional_gradient_sliding(
            f,
            grad!,
            hessian!,
            lmo,
            copy(x0),
            epsilon=1e-8,
            inner_epsilon=0.0,
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

        # with tau=0, the subproblems end through the inner tolerance
        result = FrankWolfe.second_order_conditional_gradient_sliding(
            f,
            grad!,
            hessian!,
            lmo,
            copy(x0),
            epsilon=1e-7,
            tau=0.0,
        )
        @test result.status == FrankWolfe.STATUS_OPTIMAL

        @test_throws ArgumentError FrankWolfe.second_order_conditional_gradient_sliding(
            f,
            grad!,
            hessian!,
            lmo,
            copy(x0),
            tau=-1.0,
        )
        @test_throws ArgumentError FrankWolfe.second_order_conditional_gradient_sliding(
            f,
            grad!,
            hessian!,
            lmo,
            copy(x0),
            tau=NaN,
        )
    end
end

end # module
