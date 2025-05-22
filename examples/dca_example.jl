using FrankWolfe
using LinearAlgebra
using Plots

# Example for the DcAFW algorithm
# We want to minimize f(x) - g(x)
# where f and g are convex functions.

# Let f(x) = 0.5 * ||x - a||^2
# Let g(x) = 0.5 * ||x - b||^2 (we want to subtract this, so effectively maximizing distance from b while minimizing distance to a)
# The overall objective is phi(x) = 0.5 * ||x - a||^2 - 0.5 * ||x - b||^2
# phi(x) = 0.5 * (x'x - 2a'x + a'a) - 0.5 * (x'x - 2b'x + b'b)
# phi(x) = (b-a)'x + 0.5 * (a'a - b'b)
# This is a linear function. Minimizing it over a compact set will pick a vertex.

const n = 1000
const a = randn(n)
const b = randn(n)

# f(x) = 0.5 * ||x-a||^2
function f(x)
    return 0.5 * norm(x .- a)^2
end

function grad_f!(storage, x)
    @. storage = x - a
    return nothing
end

# g(x) = 0.5 * ||x-b||^2
function g(x)
    return 0.5 * norm(x .- b)^2
end

function grad_g!(storage, x)
    @. storage = x - b
    return nothing
end

# Objective function for verification
function phi(x)
    return f(x) - g(x)
end

function main()
    # Feasible region: Probability Simplex
    lmo = FrankWolfe.ProbabilitySimplexOracle(1.0) # sum(x_i) = 1, x_i >= 0

    # Initial point: a vertex of the simplex
    x0 = FrankWolfe.compute_extreme_point(lmo, randn(n))

    println("Running DcAFW example...")
    println("Target function: 0.5 * ||x-a||^2 - 0.5 * ||x-b||^2")
    println("a = $a")
    println("b = $b")
    println("Feasible region: Probability Simplex")

    # Test with a very simple problem: min (b-a)'x
    # The LMO for DcAFW's subproblem uses grad_f(X_tk) - grad_g(xt)
    # grad_f(X_tk) - grad_g(xt) = (X_tk - a) - (xt - b)
    # The algorithm should converge to a vertex of the simplex.
    # Specifically, the one that minimizes (b-a)'x.
    # So, the LMO direction for finding the true optimum is -(b-a), or (a-b).
    # Let's find this true optimum directly.
    true_obj_grad = b .- a # Gradient of (b-a)'x is (b-a)
    # We want to minimize <b-a, x>. So, LMO computes argmin <b-a, x> = argmax <-(b-a),x> = argmax <a-b, x>
    s_opt = FrankWolfe.compute_extreme_point(lmo, true_obj_grad) # Corresponds to min <b-a, x>
    opt_val = phi(s_opt)
    println("Theoretical optimal vertex for linear objective (b-a)'x: $s_opt")
    println("Theoretical optimal value phi(s_opt): $opt_val")



    @time x_final, primal_final, traj_data, dca_gap_final, iterations = FrankWolfe.dcafw(
        f,
        grad_f!,
        g,
        grad_g!,
        lmo,
        x0,
        max_iteration=50, # Outer iterations
        max_inner_iteration=100, # Inner iterations
        epsilon=1e-5, # Inner loop tolerance
        verbose=true,
        trajectory=true,
        print_iter=5,
        memory_mode=FrankWolfe.InplaceEmphasis()
    )

    println("\nDcAFW finished.")
    println("Outer iterations: $iterations")
    println("Final solution x: $x_final")
    println("Final objective value f(x) - g(x): $primal_final") 
    println("Final objective value phi(x_final) by direct eval: $(phi(x_final))")
    println("Theoretical optimal value phi(s_opt): $opt_val")
    println("Difference from theoretical optimum: $(abs(primal_final - opt_val))")
    println("Final DCA Gap (from last inner loop): $dca_gap_final")

    # Check if solution is a vertex (for this problem, it should be)
    if any(xi -> !(isapprox(xi, 0.0, atol=1e-4) || isapprox(xi, 1.0, atol=1e-4)), x_final)
        if sum(isapprox.(x_final, 0.0, atol=1e-4) .| isapprox.(x_final, 1.0, atol=1e-4)) != 1
             println("Warning: Solution might not be a standard basis vector (vertex of probability simplex).")
        else
            println("Solution appears to be a vertex of the probability simplex.")
        end
    else
        println("Solution appears to be a vertex of the probability simplex.")
    end

    # Plotting trajectory (optional)
    # if trajectory && !isempty(traj_data)
    #     p_primal = plot([td.primal for td in traj_data], label="Primal f(x)-g(x)", xlabel="Outer Iteration", ylabel="Objective Value", yscale=:log10)
    #     p_dcagap = plot([td.dual_gap for td in traj_data], label="DCA Gap (inner)", xlabel="Outer Iteration", ylabel="Gap", yscale=:log10)
    #     display(p_primal)
    #     display(p_dcagap)
    # end

end

main() 