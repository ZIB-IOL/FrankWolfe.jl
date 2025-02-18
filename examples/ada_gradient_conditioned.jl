using FrankWolfe
using LinearAlgebra
using Random

max_iter = Int(1e5)
print_iter = max_iter // 20
epsilon = 1e-10

n = 1000
s = 42
Random.seed!(s)

# Create test problem with controlled condition number
const condition_number = 10000.0  # Much better than random conditioning
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

########## SOLVING

# Initial point
x0 = 10 * rand(n)

println("Testing Adaptive Gradient Descent (variant 1)")
println("============================================")

x1, f1, hist1 = FrankWolfe.adaptive_gradient_descent(
    f,
    grad!,
    x0;
    step0 = 0.1,
    max_iterations = max_iter,
    print_iter = print_iter,
    epsilon = epsilon,
    verbose = true
)

println("\nFinal objective value: $(f1)")
println("Final gradient norm: $(norm(grad!(similar(x0), x1)))")

println("\nTesting Adaptive Gradient Descent (variant 2)")
println("============================================")

x2, f2, hist2 = FrankWolfe.adaptive_gradient_descent2(
    f,
    grad!,
    x0;
    step0 = 0.1,
    max_iterations = max_iter,
    print_iter = print_iter,
    epsilon = epsilon,
    verbose = true
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