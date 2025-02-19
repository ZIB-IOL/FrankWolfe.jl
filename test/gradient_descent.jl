using LinearAlgebra
using Random
using Test
using FrankWolfe
using ProximalOperators

n = 100
k = Int(1e4)
print_iter = k // 10

s = 42
Random.seed!(s)

# Create test problem with controlled condition number
const condition_number = 1000.0  # Much better than random conditioning
const matrix = begin
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
            step0 = convert(T, 0.1),
            max_iterations = k,
            print_iter = print_iter,
            epsilon = target_tolerance,
            memory_mode = FrankWolfe.InplaceEmphasis(),
            verbose = true
        )
        
        # Test second variant
        x2, f2, hist2 = FrankWolfe.adaptive_gradient_descent2(
            f,
            grad!,
            convert.(T, x0);
            step0 = convert(T, 0.1),
            max_iterations = k,
            print_iter = print_iter,
            epsilon = target_tolerance,
            memory_mode = FrankWolfe.InplaceEmphasis(),
            verbose = true
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
            epsilon = target_tolerance,
            memory_mode = FrankWolfe.InplaceEmphasis(),
            print_iter = print_iter,
            verbose = true
        )
        
        # Test with OutplaceEmphasis
        x_outplace, f_outplace, _ = FrankWolfe.adaptive_gradient_descent(
            f,
            grad!,
            x0;
            epsilon = target_tolerance,
            memory_mode = FrankWolfe.OutplaceEmphasis(),
            print_iter = print_iter,
            verbose = true
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
            callback = callback,
            max_iterations = 10,
            print_iter = 1,
            verbose = true
        )
        
        # Test that callback was called and stored states
        @test length(history) == 10
        @test all(state -> length(state) == 5, history)
        @test all(state -> state[1] ≤ 10, history) # iteration numbers
        @test all(state -> typeof(state[2]) <: Real, history) # objective values
        @test issorted([state[1] for state in history]) # iterations are ordered
    end

    @testset "Projection Operators" begin
        @testset "L1 ball projection" begin
            # Test basic functionality
            x_test = [1.0, -2.0, 0.5, -0.1]
            τ = 1.0
            result = FrankWolfe.proj_l1_ball(x_test, τ)
            
            # Result should have L1 norm ≤ τ
            @test sum(abs.(result)) ≤ τ + 1e-10
            
            # Test with zero input
            @test all(FrankWolfe.proj_l1_ball(zeros(5), 1.0) .== 0.0)
            
            # Test with negative radius
            @test_throws DomainError FrankWolfe.proj_l1_ball(x_test, -1.0)
            
            # Test with zero radius
            @test all(FrankWolfe.proj_l1_ball(ones(5), 0.0) .== 0.0)
            
            # Test preservation of signs
            x_signs = sign.(x_test)
            result_signs = sign.(result)
            @test all(x_signs[abs.(result) .> 1e-10] .== result_signs[abs.(result) .> 1e-10])
            
            # Test for NaN outputs
            @test !any(isnan.(FrankWolfe.proj_l1_ball(x_test, τ)))
            @test !any(isnan.(FrankWolfe.proj_l1_ball(randn(100), 1.0)))
            
            # Test with large random input
            x_large = randn(1000)
            τ_large = 5.0
            result_large = FrankWolfe.proj_l1_ball(x_large, τ_large)
            @test sum(abs.(result_large)) ≤ τ_large + 1e-10
        end

        @testset "Probability simplex projection" begin
            # Test basic functionality
            x_test = [0.5, 0.8, -0.2, 0.4]
            result = FrankWolfe.proj_probability_simplex(x_test)
            
            # Result should sum to 1 and be non-negative
            @test sum(result) ≈ 1.0 atol=1e-10
            @test all(result .>= 0)
            
            # Test with zero input
            @test sum(FrankWolfe.proj_probability_simplex(zeros(5))) ≈ 1.0
            
            # Test with all negative input
            @test all(FrankWolfe.proj_probability_simplex(-ones(5)) .≈ 0.2)
            
            # Test for NaN outputs
            @test !any(isnan.(FrankWolfe.proj_probability_simplex(x_test)))
        end

        @testset "Unit simplex projection" begin
            # Test basic functionality
            x_test = [0.5, 0.8, -0.2, 0.4]
            τ = 2.0
            result = FrankWolfe.proj_unit_simplex(x_test, τ)
            
            # Result should sum to ≤ τ and be non-negative
            @test sum(result) ≤ τ + 1e-10
            @test all(result .>= 0)
            
            # Test case where sum should be exactly τ
            x_large = [1.0, 2.0, 3.0, 4.0]  # Sum > τ
            result_large = FrankWolfe.proj_unit_simplex(x_large, τ)
            @test sum(result_large) ≈ τ atol=1e-10
            
            # Test case where sum should be less than τ
            x_small = [0.2, 0.3, 0.1, 0.1]  # Sum < τ
            result_small = FrankWolfe.proj_unit_simplex(x_small, τ)
            @test all(result_small .≈ x_small)  # Should remain unchanged
            
            # Test with negative τ
            @test_throws DomainError FrankWolfe.proj_unit_simplex(x_test, -1.0)
            
            # Test with zero τ
            @test all(FrankWolfe.proj_unit_simplex(x_test, 0.0) .== 0.0)
            
            # Test for NaN outputs
            @test !any(isnan.(FrankWolfe.proj_unit_simplex(x_test, τ)))
        end

        @testset "Box projection" begin
            # Test basic functionality
            x_test = [0.5, 1.2, -0.2, 0.8]
            τ = 1.0
            result = FrankWolfe.proj_box(x_test, τ)
            
            # Result should be in [0,τ]
            @test all(0 .<= result .<= τ)
            
            # Test with negative τ
            @test_throws DomainError FrankWolfe.proj_box(x_test, -1.0)
            
            # Test with zero τ
            @test all(FrankWolfe.proj_box(x_test, 0.0) .== 0.0)
            
            # Test for NaN outputs
            @test !any(isnan.(FrankWolfe.proj_box(x_test, τ)))
        end
    end

    @testset "Proximal variant" begin
        x0 = rand(n)
        target_tolerance = 1e-8
        
        # Test with identity proximal operator (should match regular variant)
        x_id, f_id, _ = FrankWolfe.proximal_adaptive_gradient_descent(
            f,
            grad!,
            x0;
            epsilon = target_tolerance,
            print_iter = print_iter,
            verbose = true
        )
        
        # Test with L1 proximal operator
        x_l1, f_l1, _ = FrankWolfe.proximal_adaptive_gradient_descent(
            f,
            grad!,
            x0;
            prox = ProximalOperators.IndBallL1(1.0),
            epsilon = target_tolerance,
            print_iter = print_iter,
            verbose = true
        )
        
        # Identity proximal operator should give same result as regular variant
        x_reg, f_reg, _ = FrankWolfe.adaptive_gradient_descent(
            f,
            grad!,
            x0;
            epsilon = target_tolerance,
            print_iter = print_iter,
            verbose = true
        )

        @testset "Comparison with FW variants" begin
            @testset "L1-ball comparison" begin
                println("** L1-ball comparison")
                x0 = FrankWolfe.compute_extreme_point(FrankWolfe.LpNormLMO{Float64,1}(1.0), zeros(n));
                x_fw_l1, _ = FrankWolfe.blended_pairwise_conditional_gradient(
                    f,
                    grad!,
                    FrankWolfe.LpNormLMO{Float64,1}(1.0),
                    line_search=FrankWolfe.Secant(),
                    x0;
                    epsilon = target_tolerance,
                    max_iteration = k,
                    print_iter = print_iter,
                    verbose = true
                )
                f_fw_l1 = f(x_fw_l1)
                @test abs(f_l1 - f_fw_l1) ≤ target_tolerance * 10
            end

            @testset "Probability simplex comparison" begin
                println("** Probability simplex comparison")
                x_prox_prob, f_prox_prob, _ = FrankWolfe.proximal_adaptive_gradient_descent(
                    f,
                    grad!,
                    x0;
                    prox = ProximalOperators.IndSimplex(1.0),
                    epsilon = target_tolerance,
                    print_iter = print_iter,
                    verbose = true
                )

                x0 = FrankWolfe.compute_extreme_point(FrankWolfe.ProbabilitySimplexOracle(1.0), zeros(n));
                x_fw_prob, _ = FrankWolfe.blended_pairwise_conditional_gradient(
                    f,
                    grad!,
                    FrankWolfe.ProbabilitySimplexOracle(1.0),
                    line_search=FrankWolfe.Secant(),
                    x0;
                    epsilon = target_tolerance,
                    max_iteration = k,
                    print_iter = print_iter,
                    verbose = true
                )
                f_fw_prob = f(x_fw_prob)
                @test abs(f_prox_prob - f_fw_prob) ≤ target_tolerance * 10
            end

            @testset "Unit simplex comparison" begin
                println("** Unit simplex comparison")
                τ_unit = 2.0
                # dense test
                x0 = Vector(FrankWolfe.compute_extreme_point(FrankWolfe.UnitSimplexOracle(τ_unit), zeros(n)));
                x_prox_unit, f_prox_unit, _ = FrankWolfe.proximal_adaptive_gradient_descent(
                    f,
                    grad!,
                    x0;
                    prox = (x, t) -> FrankWolfe.proj_unit_simplex(x, τ_unit),
                    epsilon = target_tolerance,
                    print_iter = print_iter,
                    verbose = true
                )
                x_fw_unit, _ = FrankWolfe.blended_pairwise_conditional_gradient(
                    f,
                    grad!,
                    FrankWolfe.UnitSimplexOracle(τ_unit),
                    line_search=FrankWolfe.Secant(),
                    x0;
                    epsilon = target_tolerance,
                    max_iteration = k,
                    print_iter = print_iter,
                    verbose = true
                )
                f_fw_unit = f(x_fw_unit)
                @test abs(f_prox_unit - f_fw_unit) ≤ target_tolerance * 10
            end

            @testset "Box comparison" begin
                println("** Box comparison")
                τ_box = 1.0
                x_prox_box, f_prox_box, _ = FrankWolfe.proximal_adaptive_gradient_descent(
                    f,
                    grad!,
                    x0;
                    prox = (x, t) -> FrankWolfe.proj_box(x, τ_box),
                    epsilon = target_tolerance,
                    print_iter = print_iter,
                    verbose = true
                )
                x0 = FrankWolfe.compute_extreme_point(FrankWolfe.ScaledBoundLInfNormBall(-τ_box * zeros(n), τ_box * ones(n)), zeros(n));
                x_fw_box, _ = FrankWolfe.blended_pairwise_conditional_gradient(
                    f,
                    grad!,
                    FrankWolfe.ScaledBoundLInfNormBall(-τ_box * zeros(n), τ_box * ones(n)),
                    line_search=FrankWolfe.Secant(),
                    x0;
                    epsilon = target_tolerance,
                    max_iteration = k,
                    print_iter = print_iter,
                    verbose = true
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
            epsilon = target_tolerance,
            print_iter = print_iter,
            verbose = true
        )
        
        # Test with L1 proximal operator
        x_l1, f_l1, _ = FrankWolfe.proximal_adaptive_gradient_descent(
            f,
            grad!,
            x0;
            prox = (x, t) -> FrankWolfe.proj_l1_ball(x, 1.0),
            epsilon = target_tolerance,
            print_iter = print_iter,
            verbose = true
        )
        
        # Identity proximal operator should give same result as regular variant
        x_reg, f_reg, _ = FrankWolfe.adaptive_gradient_descent(
            f,
            grad!,
            x0;
            epsilon = target_tolerance,
            print_iter = print_iter,
            verbose = true
        )

        @testset "Comparison with FW variants" begin
            @testset "L1-ball comparison" begin
                println("** L1-ball comparison")
                x0 = FrankWolfe.compute_extreme_point(FrankWolfe.LpNormLMO{Float64,1}(1.0), zeros(n));
                x_fw_l1, _ = FrankWolfe.blended_pairwise_conditional_gradient(
                    f,
                    grad!,
                    FrankWolfe.LpNormLMO{Float64,1}(1.0),
                    line_search=FrankWolfe.Secant(),
                    x0;
                    epsilon = target_tolerance,
                    max_iteration = k,
                    print_iter = print_iter,
                    verbose = true
                )
                f_fw_l1 = f(x_fw_l1)
                @test abs(f_l1 - f_fw_l1) ≤ target_tolerance * 10
            end

            @testset "Probability simplex comparison" begin
                println("** Probability simplex comparison")
                x_prox_prob, f_prox_prob, _ = FrankWolfe.proximal_adaptive_gradient_descent(
                    f,
                    grad!,
                    x0;
                    prox = (x, t) -> FrankWolfe.proj_probability_simplex(x),
                    epsilon = target_tolerance,
                    print_iter = print_iter,
                    verbose = true
                )

                x0 = FrankWolfe.compute_extreme_point(FrankWolfe.ProbabilitySimplexOracle(1.0), zeros(n));
                x_fw_prob, _ = FrankWolfe.blended_pairwise_conditional_gradient(
                    f,
                    grad!,
                    FrankWolfe.ProbabilitySimplexOracle(1.0),
                    line_search=FrankWolfe.Secant(),
                    x0;
                    epsilon = target_tolerance,
                    max_iteration = k,
                    print_iter = print_iter,
                    verbose = true
                )
                f_fw_prob = f(x_fw_prob)
                @test abs(f_prox_prob - f_fw_prob) ≤ target_tolerance * 10
            end

            @testset "Unit simplex comparison" begin
                println("** Unit simplex comparison")
                τ_unit = 2.0
                # dense test
                x0 = Vector(FrankWolfe.compute_extreme_point(FrankWolfe.UnitSimplexOracle(τ_unit), zeros(n)));
                x_prox_unit, f_prox_unit, _ = FrankWolfe.proximal_adaptive_gradient_descent(
                    f,
                    grad!,
                    x0;
                    prox = (x, t) -> FrankWolfe.proj_unit_simplex(x, τ_unit),
                    epsilon = target_tolerance,
                    print_iter = print_iter,
                    verbose = true
                )
                x_fw_unit, _ = FrankWolfe.blended_pairwise_conditional_gradient(
                    f,
                    grad!,
                    FrankWolfe.UnitSimplexOracle(τ_unit),
                    line_search=FrankWolfe.Secant(),
                    x0;
                    epsilon = target_tolerance,
                    max_iteration = k,
                    print_iter = print_iter,
                    verbose = true
                )
                f_fw_unit = f(x_fw_unit)
                @test abs(f_prox_unit - f_fw_unit) ≤ target_tolerance * 10
            end

            @testset "Box comparison" begin
                println("** Box comparison")
                τ_box = 1.0
                x_prox_box, f_prox_box, _ = FrankWolfe.proximal_adaptive_gradient_descent(
                    f,
                    grad!,
                    x0;
                    prox = (x, t) -> FrankWolfe.proj_box(x, τ_box),
                    epsilon = target_tolerance,
                    print_iter = print_iter,
                    verbose = true
                )
                x0 = FrankWolfe.compute_extreme_point(FrankWolfe.ScaledBoundLInfNormBall(-τ_box * zeros(n), τ_box * ones(n)), zeros(n));
                x_fw_box, _ = FrankWolfe.blended_pairwise_conditional_gradient(
                    f,
                    grad!,
                    FrankWolfe.ScaledBoundLInfNormBall(-τ_box * zeros(n), τ_box * ones(n)),
                    line_search=FrankWolfe.Secant(),
                    x0;
                    epsilon = target_tolerance,
                    max_iteration = k,
                    print_iter = print_iter,
                    verbose = true
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