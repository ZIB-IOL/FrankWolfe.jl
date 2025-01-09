problems = ["OEDP_A", "OEDP_D", "Nuclear", "Birkhoff", "QuadraticProbSimplex", "Spectrahedron", "IllConditionedQuadratic"] 
dimensions = collect(100:100:1000) # square root of the dimension
seeds = [0] # [1,2,3,4,5]
linesearches = ["Secant", "SecantBT", "Adaptive", "BacktrackingSecant", "AdaptiveSecant", "AdaptiveZeroSecant"] #"BacktrackingSecant 

for line_search in linesearches
    for problem in problems
        for dim in dimensions
            for seed in seeds
                @show line_search, problem, dim, seed
                run(`sbatch -A optimi -J Secant jobs.sbatch $line_search $problem $dim $seed`) # CB
            end
        end
    end
end
