seeds = [0] # [1,2,3,4,5]
seeds = [1,2,3,4,5]
#linesearches = ["Secant", "SecantBT", "Adaptive", "BacktrackingSecant", "AdaptiveSecant", "AdaptiveZeroSecant", "Secant_3", "Secant_5", "Secant_7", "Secant_12"] #"BacktrackingSecant 

linesearches = ["Backtracking"]  #"Monotonic", "Agnostic", "Backtracking", "Secant", "Monotonic", "Agnostic", "Adaptive", "AdaptiveZero", "Goldenratio", 
Fw_variant = "BPCG"

#Fw_variant = "Vanilla"
# problems = ["Nuclear", "Spectrahedron"]
#linesearches = ["Secant", "Adaptive", "Monotonic","Goldenratio","Agnostic","AdaptiveZero"]


problems = ["Spectrahedron"] 
dimensions = collect(100:100:1000) # square root of the dimension
#dimensions = collect(100:100:300)

for line_search in linesearches
    for problem in problems
        for dim in dimensions
            for seed in seeds
                @show line_search, problem, dim, seed
                run(`sbatch -A optimi -J SeSpec jobs.sbatch $line_search $problem $dim $seed $Fw_variant`) # CB
            end
        end
    end
end


problems = ["Nuclear", "Birkhoff", "QuadraticProbSimplex"] 
dimensions = collect(50:50:300) # square root of the dimension
#dimensions = collect(100:100:300)

for line_search in linesearches
    for problem in problems
        for dim in dimensions
            for seed in seeds
                @show line_search, problem, dim, seed
                run(`sbatch -A optimi -J SeBNQ jobs.sbatch $line_search $problem $dim $seed $Fw_variant`) # CB
            end
        end
    end
end

problems = ["IllConditionedQuadratic"]
dimensions = collect(500:500:5000)
#dimensions = collect(500:500:2000)

for line_search in linesearches
    for problem in problems
        for dim in dimensions
            for seed in seeds
                @show line_search, problem, dim, seed
                run(`sbatch -A optimi -J SeIll jobs.sbatch $line_search $problem $dim $seed $Fw_variant`) # CB
            end
        end
    end
end
=#
problems = ["OEDP_A", "OEDP_D"] #
dimensions = collect(100:100:1000)
#dimensions = collect(500:500:2000)

for line_search in linesearches
    for problem in problems
        for dim in dimensions
            for seed in seeds
                @show line_search, problem, dim, seed
                run(`sbatch -A optimi -J SeOpt jobs.sbatch $line_search $problem $dim $seed $Fw_variant`) # CB
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
