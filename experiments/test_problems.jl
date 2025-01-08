using FrankWolfe
using Random
using LinearAlgebra
using Test

include("solve_problems.jl")

seed = 1
m = 100
problems = ["OEDP_A", "OEDP_D", "Nuclear", "Birkhoff", "QuadraticProbSimplex", "Spectrahedron", "IllConditionedQuadratic"] 

@testset "Testing the examples" begin

    for problem in problems
        @testset "$problem" begin

            _, primal_s, dual_gap_s = solve_problems(seed, m, problem, LS_ONLY_SECANT, write=false)
            _, primal_s_bt, dual_gap_s_bt = solve_problems(seed, m, problem, LS_SECANT_WITH_BACKTRACKING, write=false)
            _, primal_a, dual_gap_a = solve_problems(seed, m, problem, LS_ADAPTIVE, write=false)
            _, primal_bt_s, dual_gap_bt_s = solve_problems(seed, m, problem, LS_BACKTRACKING_AND_SECANT, write=false)


            @test primal_s >= primal_s_bt - dual_gap_s_bt
            @test primal_s_bt >= primal_a - dual_gap_a
            @test primal_a >= primal_bt_s - dual_gap_bt_s
            @test primal_bt_s >= primal_s - dual_gap_s
        end
    end

    #=@testset "Optimal Design A" begin
        f, grad!, lmo, x0, active_set, domain_oracle = build_optimal_design(seed, m, criterion="A")

        primals = []
        dual_gaps = []

        for ls in [LS_ONLY_SECANT, LS_BACKTRACKING_AND_SECANT, LS_ADAPTIVE, LS_SECANT_WITH_BACKTRACKING]
            _, primal, dual_gap = solve_problems(seed, m, "", ls_variant; time_limit=3600, write=true, verbose=true, FW_variant="BPCG")
        end
    end

    @testset "Optimal Design D" begin
    end

    @testset "Nuclear Norm" begin
    end

    @testset "Birkhoff" begin
    end

    @testset "Spectrahedron" begin
    end

    @testset "Simple Quadratic" begin
    end

    @testset "Ill Conditioned Quadratic" begin
    end =#
end