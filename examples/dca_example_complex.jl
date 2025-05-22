# Complex DCA (Difference of Convex Algorithm) with Frank-Wolfe Example
#
# This example demonstrates the DcAFW algorithm on a more complex non-trivial function pair.
# Unlike the simple example which reduces to a linear objective, this example uses:
#
# f(x) = 0.5 * x^T A x + a^T x + exp(sum(c .* x) / n) / n
#        ^^^^^^^^^^^^^^^^   ^^^^   ^^^^^^^^^^^^^^^^^^^^^^^^
#        quadratic term   linear   exponential term (convex)
#
# g(x) = 0.5 * x^T B x + b^T x + 0.1 * sum(log(1 + exp(d .* x)))
#        ^^^^^^^^^^^^^^^^   ^^^^   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#        quadratic term   linear   scaled logistic loss (convex)
#
# Objective: minimize φ(x) = f(x) - g(x) over L1 ball with radius 10
#
# The example:
# 1. Sets up complex convex functions f and g with their gradients
# 2. Runs DcAFW to minimize f(x) - g(x)
# 3. Compares performance with vanilla Frank-Wolfe on f(x) only
# 4. Generates trajectory plots showing convergence behavior
# 5. Validates constraint satisfaction and solution quality
#
# Key features demonstrated:
# - Handling non-trivial convex functions beyond quadratics
# - Exponential and logistic terms in the objective
# - Trajectory plotting and algorithm comparison
# - L1 ball constraint (promotes sparsity)
# - Proper numerical scaling to avoid overflow

using FrankWolfe
using LinearAlgebra
using Plots
using Random

include("plot_utils.jl")

Random.seed!(42)

# More complex DCA example
# We minimize f(x) - g(x) where both f and g are complex non-linear convex functions

const n = 100
const A = randn(n, n)
const A_pos = A' * A + 0.1 * I  # Make positive definite
const B = randn(n, n) 
const B_pos = B' * B + 0.1 * I  # Make positive definite
const a = randn(n)
const b = randn(n)
const c = randn(n)
const d = randn(n)

# f(x) = 0.5 * x^T A_pos x + a^T x + exp(sum(c .* x)) / n
function f(x)
    quadratic_part = 0.5 * dot(x, A_pos, x)
    linear_part = dot(a, x)
    exponential_part = exp(sum(c .* x) / n) / n  # Scaled to avoid overflow
    return quadratic_part + linear_part + exponential_part
end

function grad_f!(storage, x)
    # Gradient of quadratic part: A_pos * x
    mul!(storage, A_pos, x)
    # Gradient of linear part: a
    storage .+= a
    # Gradient of exponential part: (c / n) * exp(sum(c .* x) / n) / n
    exp_term = exp(sum(c .* x) / n) / n^2
    storage .+= c .* exp_term
    return nothing
end

# g(x) = 0.5 * x^T B_pos x + b^T x + 0.1 * sum(log(1 + exp(d .* x)))
function g(x)
    quadratic_part = 0.5 * dot(x, B_pos, x)
    linear_part = dot(b, x)
    # Logistic loss - scaled down to avoid numerical issues
    logistic_part = 0.1 * sum(log.(1 .+ exp.(d .* x)))
    return quadratic_part + linear_part + logistic_part
end

function grad_g!(storage, x)
    # Gradient of quadratic part: B_pos * x
    mul!(storage, B_pos, x)
    # Gradient of linear part: b
    storage .+= b
    # Gradient of logistic part: 0.1 * d .* sigmoid(d .* x)
    sigmoid_vals = 1 ./ (1 .+ exp.(-d .* x))
    storage .+= 0.1 * d .* sigmoid_vals
    return nothing
end

# Combined objective function for verification
function phi(x)
    return f(x) - g(x)
end

function main()
    println("Running Complex DcAFW example...")
    println("f(x) = 0.5 * x^T A x + a^T x + exp(sum(c .* x) / n) / n")
    println("g(x) = 0.5 * x^T B x + b^T x + 0.1 * sum(log(1 + exp(d .* x)))")
    println("Objective: f(x) - g(x)")
    println("Feasible region: L1 ball with radius 10")
    println()

    # Feasible region: L1 ball
    lmo = FrankWolfe.LpNormLMO{1}(10.0)

    # Initial point: a vertex of the L1 ball
    x0 = FrankWolfe.compute_extreme_point(lmo, randn(n))
    
    println("Initial point sparsity: $(sum(abs.(x0) .> 1e-6))")
    println("Initial objective value φ(x0): $(phi(x0))")
    println()

    @time x_final, primal_final, traj_data, dca_gap_final, iterations = FrankWolfe.dcafw(
        f,
        grad_f!,
        g,
        grad_g!,
        lmo,
        x0,
        max_iteration=100,     # Outer iterations
        max_inner_iteration=500,  # Inner iterations
        epsilon=1e-6,         # Inner loop tolerance
        verbose=true,
        trajectory=true,
        print_iter=10,
        memory_mode=FrankWolfe.InplaceEmphasis()
    )

    println("\nDcAFW finished.")
    println("Outer iterations: $iterations")
    println("Final solution sparsity: $(sum(abs.(x_final) .> 1e-6))")
    println("Final objective value f(x) - g(x): $primal_final")
    println("Final objective value φ(x_final) by direct eval: $(phi(x_final))")
    println("Final DCA Gap (from last inner loop): $dca_gap_final")
    
    # Check constraint satisfaction
    l1_norm = sum(abs.(x_final))
    println("L1 norm of solution: $l1_norm (constraint: ≤ 10)")
    if l1_norm <= 10.1  # Small tolerance
        println("✓ Constraint satisfied")
    else
        println("✗ Constraint violated!")
    end

    # Run a simple Frank-Wolfe baseline for comparison
    println("\n" * "="^50)
    println("Running baseline Frank-Wolfe on f(x) only for comparison...")
    
    function f_only(x)
        return f(x)
    end
    
    @time x_fw, v_fw, primal_fw, dual_gap_fw, trajectory_fw = FrankWolfe.frank_wolfe(
        f_only,
        grad_f!,
        lmo,
        x0,
        max_iteration=iterations * 10,  # Give FW more iterations to be fair
        print_iter=50,
        verbose=true,
        trajectory=true,
    )
    
    println("FW (f only) final objective: $(f_only(x_fw))")
    println("DcAFW final f value: $(f(x_final))")
    println("DcAFW final g value: $(g(x_final))")
    println("DcAFW final f-g value: $(phi(x_final))")

    # Plotting trajectories
    println("\nPlotting trajectories...")
    
    # Prepare trajectory data for DcAFW
    # The trajectory format expected by plot_trajectories:
    # Each element should have: [iteration, primal, dual, dual_gap, time, other...]
    # The trajectory data from dcafw is already in the right format: (t, primal, dual, dual_gap, time)
    dca_trajectory_formatted = []
    for td in traj_data
        # td is a tuple: (t, primal, dual, dual_gap, time)
        push!(dca_trajectory_formatted, [
            td[1],               # iteration (t)
            td[2],               # primal value
            td[3],               # dual value
            td[4],               # dual gap
            td[5],               # time
            td[4]                # extra field (duplicate dual gap)
        ])
    end
    
    # Compare with Frank-Wolfe trajectory
    fw_trajectory_formatted = []
    for td in trajectory_fw
        # td is also a tuple: (t, primal, dual, dual_gap, time)
        push!(fw_trajectory_formatted, [
            td[1],               # iteration
            td[2],               # primal value
            td[3],               # dual value
            td[4],               # dual gap
            td[5],               # time
            td[4]                # extra field
        ])
    end
    
    # Plot comparison
    plot_trajectories(
        [dca_trajectory_formatted, fw_trajectory_formatted],
        ["DcAFW (f-g)", "FW (f only)"],
        xscalelog=true,
        legend_position=:topright,
        filename="dca_example_complex.png"
    )

    return x_final, primal_final, traj_data
end

# Run the example
main() 