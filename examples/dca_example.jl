using FrankWolfe
using LinearAlgebra
using Plots

# DCA (Difference of Convex Algorithm) example with quadratic functions
# We want to minimize φ(x) = f(x) - g(x) over the probability simplex
# where f and g are convex quadratic functions.

# Problem setup: minimize φ(x) = f(x) - g(x) where:
# f(x) = 0.5 * x^T A x + a^T x + c  (convex quadratic)
# g(x) = 0.5 * x^T B x + b^T x + d  (convex quadratic)
# 
# This creates a more interesting non-convex optimization problem than simple squared distances.

const n = 500  # Reduced dimension

# Generate random positive definite matrices to ensure convexity
function generate_problem_data()
    # Create positive definite matrices
    A_raw = randn(n, n)
    A = A_raw' * A_raw + 0.1 * I  # Ensure positive definiteness
    
    B_raw = randn(n, n) 
    B = B_raw' * B_raw + 0.1 * I  # Ensure positive definitiveness
    
    # Generate random linear terms
    a = randn(n)
    b = randn(n)
    
    # Generate random constants
    c = randn()
    d = randn()
    
    return A, B, a, b, c, d
end

const A, B, a, b, c, d = generate_problem_data()

# f(x) = 0.5 * x^T A x + a^T x + c
function f(x)
    return 0.5 * dot(x, A, x) + dot(a, x) + c
end

function grad_f!(storage, x)
    mul!(storage, A, x)
    storage .+= a
    return nothing
end

# g(x) = 0.5 * x^T B x + b^T x + d  
function g(x)
    return 0.5 * dot(x, B, x) + dot(b, x) + d
end

function grad_g!(storage, x)
    mul!(storage, B, x)
    storage .+= b
    return nothing
end

# Objective function for verification
function phi(x)
    return f(x) - g(x)
end

function main()
    # Feasible region: Probability Simplex
    # lmo = FrankWolfe.ProbabilitySimplexOracle(1.0) # sum(x_i) = 1, x_i >= 0
    lmo = FrankWolfe.KSparseLMO(5, 1000.0)

    # Initial point: a vertex of the simplex
    x0 = FrankWolfe.compute_extreme_point(lmo, randn(n))

    println("Running Enhanced DcAFW example...")
    println("="^60)
    println("Problem: minimize φ(x) = f(x) - g(x) over probability simplex")
    println("f(x) = 0.5 * x^T A x + a^T x + c  (convex quadratic)")
    println("g(x) = 0.5 * x^T B x + b^T x + d  (convex quadratic)")
    println("Dimension: n = $n")
    println("Feasible region: K-sparse Polytope")
    println("="^60)
    
    # Display problem characteristics
    println("Problem characteristics:")
    println("  Matrix A condition number: $(round(cond(A), digits=2))")
    println("  Matrix B condition number: $(round(cond(B), digits=2))")
    println("  Initial objective φ(x₀): $(round(phi(x0), digits=6))")
    println("  Initial sparsity (non-zeros): $(sum(abs.(x0) .> 1e-6))")
    println()
    
    # For quadratic functions, finding theoretical optimum is more complex
    println("Note: Unlike linear objectives, quadratic difference-of-convex problems")
    println("      may have multiple local minima. DCA finds a stationary point.")
    println()



    @time x_final, primal_final, traj_data, dca_gap_final, iterations = FrankWolfe.dca_fw(
        f,
        grad_f!,
        g,
        grad_g!,
        lmo,
        x0,
        max_iteration=500, # Outer iterations
        max_inner_iteration=10000, # Inner iterations
        epsilon=1e-5, # Tolerance for DCA gap
        line_search=FrankWolfe.Secant(),
        verbose=true,
        trajectory=true,
        print_iter=10,
        memory_mode=FrankWolfe.InplaceEmphasis(),
        bpcg_subsolver=true,
        warm_start=true,
        use_dca_early_stopping=true,
    )

    println("\n" * "="^60)
    println("DcAFW Algorithm Results")
    println("="^60)
    println("Convergence:")
    println("  Outer iterations completed: $iterations") 
    println("  Final objective value φ(x): $(round(primal_final, digits=8))")
    println("  Verification φ(x) = f(x) - g(x): $(round(phi(x_final), digits=8))")
    println("  Final DCA gap: $(round(dca_gap_final, digits=8))")
    println()
    println("Solution properties:")
    println("  Final solution sparsity: $(sum(abs.(x_final) .> 1e-6))")
    println("  Solution feasibility check: sum(x) = $(round(sum(x_final), digits=8))")
    println("  All components ≥ 0: $(all(x_final .>= -1e-10))")

end

main() 