using FrankWolfe
using LinearAlgebra
using Random

max_iter = 10_000
print_iter = max_iter ÷ 100
epsilon = 1e-5

"""
Simple quadratic function f(x) = 1/2 * x'Qx + b'x
"""
function quadratic_oracle(x, Q, b)
    return 0.5 * dot(x, Q, x) + dot(b, x)
end

"""
Gradient of quadratic function ∇f(x) = Qx + b
"""
function quadratic_gradient!(storage, x, Q, b)
    mul!(storage, Q, x)
    storage .+= b
    return storage
end

# Set random seed for reproducibility.
Random.seed!(42)

# Problem dimension
n = 10000

# Generate positive definite Q matrix and random b vector
Q = rand(n, n)
Q = Q' * Q + I  # Make Q positive definite
b = rand(n)

# Create objective function and gradient
f(x) = quadratic_oracle(x, Q, b)
grad!(storage, x) = quadratic_gradient!(storage, x, Q, b)

# Initial point
x0 = 10 * rand(n)

println("Testing Adaptive Gradient Descent (variant 1)")
println("============================================")

x1, f1, hist1 = FrankWolfe.Experimental.adaptive_gradient_descent(
    f,
    grad!,
    x0;
    step0=0.1,
    max_iteration=max_iter,
    print_iter=print_iter,
    epsilon=epsilon,
    verbose=true,
)

println("\nFinal objective value: $(f1)")
println("Final gradient norm: $(norm(grad!(similar(x0), x1)))")

println("\nTesting Adaptive Gradient Descent (variant 2)")
println("============================================")

x2, f2, hist2 = FrankWolfe.Experimental.adaptive_gradient_descent2(
    f,
    grad!,
    x0;
    step0=0.1,
    max_iteration=max_iter,
    print_iter=print_iter,
    epsilon=epsilon,
    verbose=true,
)

println("\nFinal objective value: $(f2)")
println("Final gradient norm: $(norm(grad!(similar(x0), x2)))")

# Compare the two methods
println("\nComparison")
println("==========")
println("Method 1 final objective: $(f1)")
println("Method 2 final objective: $(f2)")
println("Objective difference: $(abs(f1 - f2))")
println("Solution difference norm: $(norm(x1 - x2))")
