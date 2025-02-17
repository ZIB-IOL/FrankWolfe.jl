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
            print_iter = print_iter,
            verbose = true
        )
        
        # Test that callback was called and stored states
        @test length(history) == 10
        @test all(state -> length(state) == 5, history)
        @test all(state -> state[1] ≤ 10, history) # iteration numbers
        @test all(state -> typeof(state[2]) <: Real, history) # objective values
        @test issorted([state[1] for state in history]) # iterations are ordered
    end
end 