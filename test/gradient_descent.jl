using LinearAlgebra
using Random
using Test
using FrankWolfe
using DoubleFloats

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

f(x) = dot(linear, x) + 0.5 * transpose(x) * hessian * x

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

    @testset "Proximal variant" begin
        x0 = rand(n)
        target_tolerance = 1e-8
        
        # Define a simple L1 proximal operator for testing
        # prox_l1(x, t) = sign.(x) .* max.(abs.(x) .- t, 0)

        # Test that prox_l1 operator behaves correctly
        # @testset "prox_l1 operator" begin
        #     # Test basic functionality
        #     x_test = [1.0, -2.0, 0.5, -0.1]
        #     t = 0.5
        #     result = prox_l1(x_test, t)
            
        #     # Should shrink values toward zero by t
        #     @test result ≈ [0.5, -1.5, 0.0, 0.0]
            
        #     # Test with zero input
        #     @test all(prox_l1(zeros(5), 1.0) .== 0.0)
            
        #     # Test with negative threshold
        #     @test_throws DomainError prox_l1(x_test, -1.0)
            
        #     # Test with various thresholds
        #     @test all(prox_l1(ones(5), 2.0) .== 0.0)  # Full thresholding
        #     @test prox_l1(ones(5), 0.0) ≈ ones(5)     # No thresholding
            
        #     # Test for NaN outputs
        #     @test !any(isnan.(prox_l1(x_test, t)))
        #     @test !any(isnan.(prox_l1(randn(100), 0.1)))
        # end

        
        # Project onto L1 ball with radius τ
        function proj_l1_ball(x, τ)
            # Debug: check for NaN in input
            if any(isnan.(x))
                println("Debug: Input x contains NaN values")
                println("Debug: Input x type = $(typeof(x))")
            end

            if τ < 0
                throw(DomainError(τ, "L1 ball radius must be non-negative"))
            end
            
            # Handle trivial cases
            if norm(x, 1) <= τ
                return copy(x)
            end
            if τ == 0
                return zero(x)
            end
            
            # Sort absolute values in descending order
            u = sort(abs.(x), rev=true)
            
            # Find largest k such that u_k > θ
            cumsum_u = cumsum(u)
            k = findlast(i -> u[i] > (cumsum_u[i] - τ) / i, 1:length(x))
            # println("k: $k")
            # Debug output if k is nothing
            if isnothing(k)
                println("Debug: cumsum_u = ", cumsum_u)
                println("Debug: norm(x,1) = ", norm(x,1))
            end
            
            # Compute θ
            θ = (cumsum_u[k] - τ) / k
            
            # Apply soft-thresholding
            return sign.(x) .* max.(abs.(x) .- θ, 0)
        end

        prox_l1(x, t) = begin
            return proj_l1_ball(x, 1)
        end
        
        # Test L1 projection operator
        @testset "L1 projection operator" begin
            # Test basic functionality
            x_test = [1.0, -2.0, 0.5, -0.1]
            τ = 1.0
            result = proj_l1_ball(x_test, τ)
            
            # Result should have L1 norm ≤ τ
            @test sum(abs.(result)) ≤ τ + 1e-10
            
            # Test with zero input
            @test all(proj_l1_ball(zeros(5), 1.0) .== 0.0)
            
            # Test with negative radius
            @test_throws DomainError proj_l1_ball(x_test, -1.0)
            
            # Test with zero radius
            @test all(proj_l1_ball(ones(5), 0.0) .== 0.0)
            
            # Test preservation of signs
            x_signs = sign.(x_test)
            result_signs = sign.(result)
            @test all(x_signs[abs.(result) .> 1e-10] .== result_signs[abs.(result) .> 1e-10])
            
            # Test for NaN outputs
            @test !any(isnan.(proj_l1_ball(x_test, τ)))
            @test !any(isnan.(proj_l1_ball(randn(100), 1.0)))
            
            # Test with large random input
            x_large = randn(1000)
            τ_large = 5.0
            result_large = proj_l1_ball(x_large, τ_large)
            @test sum(abs.(result_large)) ≤ τ_large + 1e-10
        end

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
            prox = prox_l1,
            epsilon = target_tolerance,
            print_iter = print_iter,
            verbose = true
        )
        
        # Test convergence
        @test norm(grad!(similar(x_id), x_id)) ≤ target_tolerance

        # needs a different error test / probably compute same problem with FW
        # @test norm(grad!(similar(x_l1), x_l1)) ≤ target_tolerance
        
        # Identity proximal operator should give same result as regular variant
        x_reg, f_reg, _ = FrankWolfe.adaptive_gradient_descent(
            f,
            grad!,
            x0;
            epsilon = target_tolerance,
            print_iter = print_iter,
            verbose = true
        )
        
        # Run Frank-Wolfe on same problem over L1 ball for comparison
        x_fw, _ = FrankWolfe.blended_pairwise_conditional_gradient(
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

        f_fw = f(x_fw)
        # Compare L1-proximal solution with FW solution
        @test abs(f_l1 - f_fw) ≤ target_tolerance * 10 # Allow slightly larger tolerance for different methods
        
        @test norm(x_id - x_reg) ≤ target_tolerance
        @test abs(f_id - f_reg) ≤ target_tolerance
        
        # L1 solution should be sparse
        @test count(x -> abs(x) > 1e-6, x_l1) < n
    end
end 