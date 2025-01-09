using FrankWolfe
using Random
using LinearAlgebra
using Test

include("solve_problems.jl")

seed = 1
m = 100
problems = ["OEDP_A", "OEDP_D", "Nuclear", "Birkhoff", "QuadraticProbSimplex", "Spectrahedron", "IllConditionedQuadratic"] #"IllConditionedQuadratic"

#problems = ["IllConditionedQuadratic"]

@testset "Testing the examples" begin

    for problem in problems
        @testset "$problem" begin

            @show problem

            _, primal_s, dual_gap_s = solve_problems(seed, m, problem, LS_ONLY_SECANT, write=false, time_limit=60)
            _, primal_s_bt, dual_gap_s_bt = solve_problems(seed, m, problem, LS_SECANT_WITH_BACKTRACKING, write=false, time_limit=60)
            _, primal_a, dual_gap_a = solve_problems(seed, m, problem, LS_ADAPTIVE, write=false, time_limit=60)
            _, primal_bt_s, dual_gap_bt_s = solve_problems(seed, m, problem, LS_BACKTRACKING_AND_SECANT, write=false, time_limit=60)
            _, primal_a_s, dual_gap_a_s = solve_problems(seed, m, problem, LS_ADAPTIVE_AND_SECANT, write=false, time_limit=60)
            _, primal_a_z_s, dual_gap_a_z_s = solve_problems(seed, m, problem, LS_ADAPTIVE_ZERO_AND_SECANT, write=false, time_limit=60)


            @test primal_s >= primal_s_bt - dual_gap_s_bt
            @test primal_s_bt >= primal_a - dual_gap_a
            @test primal_a >= primal_bt_s - dual_gap_bt_s
            @test primal_bt_s >= primal_a_s - dual_gap_a_s
            @test primal_a_s >= primal_a_z_s - dual_gap_a_z_s
            @test primal_a_z_s >= primal_s - dual_gap_s
        end
    end
end