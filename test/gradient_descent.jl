using LinearAlgebra
using Random
using Test
using FrankWolfe
using ProximalOperators
using DoubleFloats

n = 100
k = Int(1e4)
print_iter = k // 10

s = 42
Random.seed!(s)

# Create test problem with controlled condition number
const condition_number = 1000.0  # Much better than random conditioning
const matrix = let
    # Create orthogonal matrix
    Q = qr(randn(n, n)).Q
    # Create diagonal matrix with controlled condition number
    λ_max = 1.0
    λ_min = λ_max / condition_number
    Λ = Diagonal(range(λ_min, λ_max, length=n))
    # Final matrix with controlled conditioning
    Q * sqrt(Λ)
end
const hessian = transpose(matrix) * matrix
const linear = rand(n)

f(x) = dot(linear, x) + 0.5 * dot(x, hessian, x)

function grad!(storage, x)
    return storage .= linear + hessian * x
end

const L = eigmax(hessian)

# Compute optimal solution using direct solve for testing
const x_opt = -hessian \ linear
const f_opt = f(x_opt)

println("\nTesting adaptive gradient descent algorithms...\n")
println("Test instance statistics:")
println("------------------------")
println("Dimension n: $n")
println("Lipschitz constant L: $L")
println("Optimal objective value f*: $f_opt")
println("Optimal solution norm: $(norm(x_opt))")
println("Problem condition number: $(eigmax(hessian)/eigmin(hessian))")
println()

@testset "Adaptive Gradient Descent" begin
    @testset "Type $T" for T in (Float64, Double64)
        x0 = 10 * rand(T, n)
        target_tolerance = convert(T, 1e-8)

        # Test first variant
        x1, f1, hist1 = FrankWolfe.adaptive_gradient_descent(
            f,
            grad!,
            convert.(T, x0);
            step0=convert(T, 0.1),
            max_iteration=k,
            print_iter=print_iter,
            epsilon=target_tolerance,
            memory_mode=FrankWolfe.InplaceEmphasis(),
            verbose=true,
        )

        # Test second variant
        x2, f2, hist2 = FrankWolfe.adaptive_gradient_descent2(
            f,
            grad!,
            convert.(T, x0);
            step0=convert(T, 0.1),
            max_iteration=k,
            print_iter=print_iter,
            epsilon=target_tolerance,
            memory_mode=FrankWolfe.InplaceEmphasis(),
            verbose=true,
        )

        # Test convergence to optimal solution
        @test norm(grad!(similar(x1), x1)) ≤ target_tolerance
        @test norm(grad!(similar(x2), x2)) ≤ target_tolerance

        # Test objective values
        @test abs(f1 - f_opt) ≤ target_tolerance * L
        @test abs(f2 - f_opt) ≤ target_tolerance * L

        # Test solution accuracy
        @test norm(x1 - x_opt) ≤ sqrt(target_tolerance)
        @test norm(x2 - x_opt) ≤ sqrt(target_tolerance)
    end

    @testset "Memory modes" begin
        x0 = rand(n)
        target_tolerance = 1e-8

        # Test with InplaceEmphasis
        x_inplace, f_inplace, _ = FrankWolfe.adaptive_gradient_descent(
            f,
            grad!,
            x0;
            epsilon=target_tolerance,
            memory_mode=FrankWolfe.InplaceEmphasis(),
            print_iter=print_iter,
            verbose=true,
        )

        # Test with OutplaceEmphasis
        x_outplace, f_outplace, _ = FrankWolfe.adaptive_gradient_descent(
            f,
            grad!,
            x0;
            epsilon=target_tolerance,
            memory_mode=FrankWolfe.OutplaceEmphasis(),
            print_iter=print_iter,
            verbose=true,
        )

        # Results should be the same regardless of memory mode
        @test norm(x_inplace - x_outplace) ≤ target_tolerance
        @test abs(f_inplace - f_outplace) ≤ target_tolerance
    end

    @testset "Callback functionality" begin
        x0 = rand(n)
        history = []

        callback(state) = push!(history, state)

        _, _, _ = FrankWolfe.adaptive_gradient_descent(
            f,
            grad!,
            x0;
            callback=callback,
            max_iteration=10,
            print_iter=1,
            verbose=true,
        )

        # Test that callback was called and stored states
        @test length(history) == 10
        @test all(state -> length(state) == 5, history)
        @test all(state -> state[1] ≤ 10, history) # iteration numbers
        @test all(state -> typeof(state[2]) <: Real, history) # objective values
        @test issorted([state[1] for state in history]) # iterations are ordered
    end

    @testset "Proximal variant" begin
        x0 = rand(n)
        target_tolerance = 1e-8

        # Test with identity proximal operator (should match regular variant)
        x_id, f_id, _ = FrankWolfe.proximal_adaptive_gradient_descent(
            f,
            grad!,
            x0;
            epsilon=target_tolerance,
            verbose=true,
        )

        # Test with L1 proximal operator
        x_l1, f_l1, _ = FrankWolfe.proximal_adaptive_gradient_descent(
            f,
            grad!,
            x0,
            ProximalOperators.IndBallL1(1.0);
            epsilon=target_tolerance,
            max_iteration=k,
            verbose=true,
        )

        # Identity proximal operator should give same result as regular variant
        x_reg, f_reg, _ =
            FrankWolfe.adaptive_gradient_descent(f, grad!, x0; epsilon=target_tolerance, verbose=true)

        @testset "Comparison with FW variants" begin
            @testset "L1-ball comparison" begin
                x0 =
                    FrankWolfe.compute_extreme_point(FrankWolfe.LpNormLMO{Float64,1}(1.0), zeros(n))
                x_fw_l1, _ = FrankWolfe.blended_pairwise_conditional_gradient(
                    f,
                    grad!,
                    FrankWolfe.LpNormLMO{Float64,1}(1.0),
                    line_search=FrankWolfe.Secant(),
                    x0;
                    epsilon=target_tolerance,
                    max_iteration=k,
                    verbose=false,
                )
                f_fw_l1 = f(x_fw_l1)
                @test abs(f_l1 - f_fw_l1) ≤ target_tolerance * 10
            end

            @testset "Probability simplex comparison" begin
                x_prox_prob, f_prox_prob, _ = FrankWolfe.proximal_adaptive_gradient_descent(
                    f,
                    grad!,
                    x0,
                    ProximalOperators.IndSimplex(1.0);
                    epsilon=target_tolerance,
                    verbose=false,
                )

                x0 = FrankWolfe.compute_extreme_point(
                    FrankWolfe.ProbabilitySimplexOracle(1.0),
                    zeros(n),
                )
                x_fw_prob, _ = FrankWolfe.blended_pairwise_conditional_gradient(
                    f,
                    grad!,
                    FrankWolfe.ProbabilitySimplexOracle(1.0),
                    line_search=FrankWolfe.Secant(),
                    x0;
                    epsilon=target_tolerance,
                    max_iteration=k,
                    verbose=false,
                )
                f_fw_prob = f(x_fw_prob)
                @test abs(f_prox_prob - f_fw_prob) ≤ target_tolerance * 10
            end
            # TODO add unit simplex comparison once it is merged in ProximalOperators

            @testset "Box comparison" begin
                τ_box = 1.0
                x_prox_box, f_prox_box, _ = FrankWolfe.proximal_adaptive_gradient_descent(
                    f,
                    grad!,
                    x0,
                    ProximalOperators.IndBox(0.0, τ_box);
                    epsilon=target_tolerance,
                    print_iter=print_iter,
                    verbose=false,
                )
                lmo_box = FrankWolfe.ScaledBoundLInfNormBall(zeros(n), τ_box * ones(n))
                x0 = FrankWolfe.compute_extreme_point(lmo_box, zeros(n))
                x_fw_box, _ = FrankWolfe.blended_pairwise_conditional_gradient(
                    f,
                    grad!,
                    lmo_box,
                    line_search=FrankWolfe.Secant(),
                    x0;
                    epsilon=target_tolerance,
                    max_iteration=k,
                    verbose=false,
                )
                f_fw_box = f(x_fw_box)
                @test abs(f_prox_box - f_fw_box) ≤ target_tolerance * 10
            end
        end

        # Test convergence for identity proximal operator
        @test norm(grad!(similar(x_id), x_id)) ≤ target_tolerance

        # Test whether identity and regular variant give same result
        @test norm(x_id - x_reg) ≤ target_tolerance
        @test abs(f_id - f_reg) ≤ target_tolerance
    end

    @testset "Proximal variant with positive orthant solution" begin

        # Create test problem with optimal solution in positive orthant
        matrix_pos = begin
            # Create orthogonal matrix 
            Q = qr(randn(n, n)).Q
            # Create diagonal matrix with controlled condition number
            λ_max = 1.0
            λ_min = λ_max / condition_number
            Λ = Diagonal(range(λ_min, λ_max, length=n))
            # Final matrix with controlled conditioning
            Q * sqrt(Λ)
        end
        hessian_pos = transpose(matrix_pos) * matrix_pos
        # Make linear term negative to push solution into positive orthant
        linear_pos = -10.0 * ones(n)

        f(x) = dot(linear_pos, x) + 0.5 * transpose(x) * hessian_pos * x

        function grad!(storage, x)
            return storage .= linear_pos + hessian_pos * x
        end

        L = eigmax(hessian_pos)

        # Compute optimal solution using direct solve
        x_opt_pos = -hessian_pos \ linear_pos
        f_opt_pos = f(x_opt_pos)

        println("\nTesting proximal gradient descent with positive orthant solution...")
        println("Test instance statistics:")
        println("------------------------")
        println("Dimension n: $n")
        println("Lipschitz constant L: $L")
        println("Optimal objective value f*: $f_opt_pos")
        println("Optimal solution norm: $(norm(x_opt_pos))")
        println("Problem condition number: $(eigmax(hessian_pos)/eigmin(hessian_pos))")
        println("Min component of optimal solution: $(minimum(x_opt_pos))")
        println()

        x0 = rand(n)
        target_tolerance = 1e-8

        # Test with identity proximal operator (should match regular variant)
        x_id, f_id, _ = FrankWolfe.proximal_adaptive_gradient_descent(
            f,
            grad!,
            x0;
            epsilon=target_tolerance,
            print_iter=print_iter,
            verbose=true,
        )

        # Test with L1 proximal operator
        x_l1, f_l1, _ = FrankWolfe.proximal_adaptive_gradient_descent(
            f,
            grad!,
            x0,
            ProximalOperators.IndBallL1(1.0);
            epsilon=target_tolerance,
        )

        # Identity proximal operator should give same result as regular variant
        x_reg, f_reg, _ =
            FrankWolfe.adaptive_gradient_descent(f, grad!, x0; epsilon=target_tolerance)

        @testset "Comparison with FW variants" begin
            @testset "L1-ball comparison" begin
                x0 =
                    FrankWolfe.compute_extreme_point(FrankWolfe.LpNormLMO{Float64,1}(1.0), zeros(n))
                x_fw_l1, _ = FrankWolfe.blended_pairwise_conditional_gradient(
                    f,
                    grad!,
                    FrankWolfe.LpNormLMO{Float64,1}(1.0),
                    line_search=FrankWolfe.Secant(),
                    x0;
                    epsilon=target_tolerance,
                    max_iteration=k,
                    print_iter=print_iter,
                    verbose=true,
                )
                f_fw_l1 = f(x_fw_l1)
                @test abs(f_l1 - f_fw_l1) ≤ target_tolerance * 10
            end

            @testset "Probability simplex comparison" begin
                x_prox_prob, f_prox_prob, _ = FrankWolfe.proximal_adaptive_gradient_descent(
                    f,
                    grad!,
                    x0,
                    FrankWolfe.ProbabilitySimplexProx();
                    epsilon=target_tolerance,
                )
                lmo_probsimplex = FrankWolfe.ProbabilitySimplexOracle(1.0)
                x0 = FrankWolfe.compute_extreme_point(lmo_probsimplex, zeros(n))
                x_fw_prob, _ = FrankWolfe.blended_pairwise_conditional_gradient(
                    f,
                    grad!,
                    lmo_probsimplex,
                    line_search=FrankWolfe.Secant(),
                    x0;
                    epsilon=target_tolerance,
                    max_iteration=k,
                )
                f_fw_prob = f(x_fw_prob)
                @test abs(f_prox_prob - f_fw_prob) ≤ target_tolerance * 10
            end

            # TODO readd unit simplex

            @testset "Box comparison" begin
                τ_box = 1.0
                x_prox_box, f_prox_box, _ = FrankWolfe.proximal_adaptive_gradient_descent(
                    f,
                    grad!,
                    x0,
                    ProximalOperators.IndBox(-τ_box, τ_box);
                    epsilon=target_tolerance,
                )
                lmo_box = FrankWolfe.ScaledBoundLInfNormBall(-τ_box * ones(n), τ_box * ones(n))
                x0 = FrankWolfe.compute_extreme_point(lmo_box, zeros(n))
                x_fw_box, _ = FrankWolfe.blended_pairwise_conditional_gradient(
                    f,
                    grad!,
                    lmo_box,
                    line_search=FrankWolfe.Secant(),
                    x0;
                    epsilon=target_tolerance,
                    max_iteration=k,
                )
                f_fw_box = f(x_fw_box)
                @test abs(f_prox_box - f_fw_box) ≤ target_tolerance * 10
            end
        end

        # Test convergence for identity proximal operator
        @test norm(grad!(similar(x_id), x_id)) ≤ target_tolerance

        # Test whether identity and regular variant give same result
        @test norm(x_id - x_reg) ≤ target_tolerance
        @test abs(f_id - f_reg) ≤ target_tolerance
    end
end
