using CSV
using DataFrames

include("solve_problems.jl")

function merge_csvs(problem, ls)
    dimensions = 100:100:1000
    seeds = 1:5
end

problems = problems = ["OEDP_A", "OEDP_D", "Nuclear", "Birkhoff", "QuadraticProbSimplex", "Spectrahedron", "IllConditionedQuadratic"] 

for problem in problems
    for ls in [LS_ONLY_SECANT, LS_ADAPTIVE, LS_BACKTRACKING_AND_SECANT, LS_SECANT_WITH_BACKTRACKING]
        merge_csvs(problem, ls)
    end
end

