seeds = [0] # [1,2,3,4,5]
linesearches = ["Secant", "SecantBT", "Adaptive", "BacktrackingSecant", "AdaptiveSecant", "AdaptiveZeroSecant", "Secant_3", "Secant_5", "Secant_7", "Secant_12"] #"BacktrackingSecant 

linesearches = ["Monotonic"]
Fw_variant = "BPCG"

# Fw_variant = "Vanilla
# problems = ["Nuclear", "Spectrahedron"]
problems = ["Nuclear", "Birkhoff", "QuadraticProbSimplex", "Spectrahedron"] 
dimensions = collect(100:100:1000) # square root of the dimension

for line_search in linesearches
    for problem in problems
        for dim in dimensions
            for seed in seeds
                @show line_search, problem, dim, seed
                run(`sbatch -A optimi -J SeBig jobs.sbatch $line_search $problem $dim $seed $Fw_variant`) # CB
            end
        end
    end
end

problems = ["OEDP_A", "OEDP_D", "IllConditionedQuadratic"]
dimensions = collect(500:500:5000)

for line_search in linesearches
    for problem in problems
        for dim in dimensions
            for seed in seeds
                @show line_search, problem, dim, seed
                run(`sbatch -A optimi -J SeSmall jobs.sbatch $line_search $problem $dim $seed $Fw_variant`) # CB
            end
        end
    end
end

problems = ["Portfolio"]
dimensions = [800, 1200, 1500]

for line_search in linesearches
    for problem in problems
        for dim in dimensions
            for seed in seeds
                @show line_search, problem, dim, seed
                run(`sbatch -A optimi -J SePort jobs.sbatch $line_search $problem $dim $seed $Fw_variant`) # CB
            end
        end
    end
end

