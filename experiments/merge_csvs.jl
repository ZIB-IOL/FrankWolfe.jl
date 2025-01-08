using CSV
using DataFrames

include("solve_problems.jl")

function merge_csvs(problem, ls)
    dimensions = collect(100:100:1000)
    seeds = collect(1:5)

    file_name = joinpath(@__DIR__, "csv/" * problem * "/" * string(ls_variant) * string(dimensions[1]) * "_" * string(seeds[1]) * ".csv")
    df = DataFrame(CSV.File(file_name))
    
    select!(df, Not(:time))
    df[!, "time"] = [3600.0]
    select!(df, Not(:fw_time))
    df[!, "fw_time"] = [3600.0]
    deleteat!(df, 1)

    for dim in dimensions
        for seed in seeds
            try
                file_name = joinpath(@__DIR__, "csv/" * problem * "/" * string(ls_variant) * string(dim^2) * "_" * string(seed) * ".csv")
                df_temp = if isfile(file_name)
                    DataFrame(CSV.File(file_name))
                else
                    error("Problem: $(problem) Line Search variant: $(string(ls_variant)) Dimension: $(dim^2) Seed: $(seed)")
                end
            catch e
                println(e)
            end

        end
    end

    file_name = joinpath(@__DIR__, "csv/" * problem * "/" * string(ls_variant) * ".csv")
    CSV.write(file_name, df, append=false)
end

problems = problems = ["OEDP_A", "OEDP_D", "Nuclear", "Birkhoff", "QuadraticProbSimplex", "Spectrahedron", "IllConditionedQuadratic"] 

for problem in problems
    for ls in [LS_ONLY_SECANT, LS_ADAPTIVE, LS_BACKTRACKING_AND_SECANT, LS_SECANT_WITH_BACKTRACKING]
        merge_csvs(problem, ls)
    end
end

