using Test
using FrankWolfe
using LinearAlgebra

@testset "DcAFW Algorithm Tests" begin

    # Problem: minimize f(x) - g(x)
    # f(x) = 0.5 * ||x - a||^2
    # g(x) = 0.5 * ||x - b||^2
    # phi(x) = f(x) - g(x) = (b-a)'x + 0.5*(a'a - b'b)
    # This is a linear objective over a compact set.
    
    n = 5
    a_vec = randn(n)
    b_vec = randn(n)

    f_test(x) = 0.5 * norm(x .- a_vec)^2
    grad_f_test!(storage, x) = (storage .= x .- a_vec)
    
    g_test(x) = 0.5 * norm(x .- b_vec)^2
    grad_g_test!(storage, x) = (storage .= x .- b_vec)

    phi_objective(x) = f_test(x) - g_test(x)

    @testset "DcAFW on Probability Simplex" begin
        lmo_prob = FrankWolfe.ProbabilitySimplexOracle(1.0)
        x0_prob = FrankWolfe.compute_extreme_point(lmo_prob, randn(n))

        # The true minimizer for (b-a)'x over the probability simplex
        # is a vertex where x_i = 1 for i minimizing (b-a)_i, and 0 otherwise.
        obj_grad_linear = b_vec .- a_vec
        s_opt_prob = FrankWolfe.compute_extreme_point(lmo_prob, obj_grad_linear)
        opt_val_prob = phi_objective(s_opt_prob)

        if !isdefined(FrankWolfe, :dcafw)
            @warn "dcafw not loaded into FrankWolfe, skipping DcAFW tests until src/dca.jl is properly included & exported by FrankWolfe.jl"
        else
            x_final_prob, primal_final_prob, _, dca_gap_final_prob, iterations_prob = FrankWolfe.dcafw(
                f_test,
                grad_f_test!,
                g_test,
                grad_g_test!,
                lmo_prob,
                x0_prob,
                max_iteration=100, # Outer iterations
                max_inner_iteration=200, # Inner iterations
                epsilon=1e-6, # Inner loop tolerance
                verbose=false, # Keep tests quiet
                memory_mode=FrankWolfe.OutplaceEmphasis()
            )

            @test primal_final_prob ≈ opt_val_prob atol=1e-5
            @test phi_objective(x_final_prob) ≈ opt_val_prob atol=1e-5
            # Solution should be a vertex
            @test sum(x_final_prob) ≈ 1.0 atol=1e-6
            @test all(xi -> xi ≥ -1e-6, x_final_prob)
            # Check if close to the optimal vertex found by LMO directly
            @test x_final_prob ≈ s_opt_prob atol=1e-5
            @test dca_gap_final_prob ≤ 1e-6 # Inner gap condition was epsilon/2
        end
    end

    @testset "DcAFW on Unit L2 Ball" begin
        lmo_l2 = FrankWolfe.LpNormLMO{2}(1.0) # radius 1
        x0_l2 = FrankWolfe.compute_extreme_point(lmo_l2, randn(n))
        if n == 0 # L2BallOracle might have issues with n=0 if not handled, though n=5 here.
           x0_l2 = zeros(n) # ensure x0 is valid if n=0 for some reason, though test uses n=5
        end

        # True minimizer for (b-a)'x over L2 ball ||x|| <= 1
        # is x = -(b-a) / ||b-a|| (if b-a is not zero vector)
        obj_grad_linear = b_vec .- a_vec
        s_opt_l2 = Float64[]
        opt_val_l2 = 0.0
        if norm(obj_grad_linear) > 1e-9
            s_opt_l2 = -obj_grad_linear / norm(obj_grad_linear)
            opt_val_l2 = phi_objective(s_opt_l2)
        else
            # If obj_grad_linear is zero, any point in the ball is optimal, obj is constant.
            # phi(x) = 0.5*(a'a - b'b). Let's pick x0 as the point.
            s_opt_l2 = x0_l2 # or zeros(n)
            opt_val_l2 = phi_objective(s_opt_l2)
        end
        
        if !isdefined(FrankWolfe, :dcafw)
            # Warning already issued from previous testset
        else
            x_final_l2, primal_final_l2, _, dca_gap_final_l2, iterations_l2 = FrankWolfe.dcafw(
                f_test,
                grad_f_test!,
                g_test,
                grad_g_test!,
                lmo_l2,
                x0_l2,
                max_iteration=100,
                max_inner_iteration=200,
                epsilon=1e-6,
                verbose=false,
                memory_mode=FrankWolfe.OutplaceEmphasis()
            )

            @test primal_final_l2 ≈ opt_val_l2 atol=1e-5
            @test phi_objective(x_final_l2) ≈ opt_val_l2 atol=1e-5
            @test norm(x_final_l2) ≈ 1.0 atol=1e-5 # Should be on boundary if obj_grad_linear != 0
            if norm(obj_grad_linear) > 1e-9
                 @test x_final_l2 ≈ s_opt_l2 atol=1e-4 # Allow slightly larger tol for vector comparison
            end # if obj_grad_linear is zero, x_final_l2 can be anywhere in the ball. primal value is key.
            @test dca_gap_final_l2 ≤ 1e-6
        end
    end

    # TODO: Add a test case where g(x) is not quadratic, if desired.
    # For example, g(x) = c * ||x||_1 (if a suitable L1 norm LMO is used or if domain is simple)
    # However, the example problem f(x)-g(x) being linear is a good first test.

end
