using CSV
using DataFrames

include("utilities.jl")

function merge_csvs(problem, ls_variant; dimensions = collect(100:100:1000))
    seeds = collect(1:5)

    file_name = joinpath(@__DIR__, "csv/" * problem * "/" * string(ls_variant) * "_" * string(dimensions[1]) * "_" * string(seeds[1]) * ".csv")
    df = DataFrame(CSV.File(file_name))
    
    select!(df, Not(:time))
    df[!, "time"] = [3600.0]
    select!(df, Not(:fw_time))
    df[!, "fw_time"] = [3600.0]
    deleteat!(df, 1)

    for dim in dimensions
        for seed in seeds
            try
                file_name = joinpath(@__DIR__, "csv/" * problem * "/" * string(ls_variant) * "_" * string(dim) * "_" * string(seed) * ".csv")
                df_temp = #if isfile(file_name)
                    DataFrame(CSV.File(file_name))
                    append!(df, df_temp)
                #else
                    #error("Problem: $(problem) Line Search variant: $(string(ls_variant)) Dimension: $(dim) Seed: $(seed)")
                #end
            catch e
                println("Problem: $(problem) Line Search variant: $(string(ls_variant)) Dimension: $(dim) Seed: $(seed)")
                println(e)
            end

        end
    end

    file_name = joinpath(@__DIR__, "csv/" * problem * "/" * string(ls_variant) * ".csv")
    CSV.write(file_name, df, append=false)
end

problems = problems = ["OEDP_A", "OEDP_D", "Nuclear", "Birkhoff", "QuadraticProbSimplex", "Spectrahedron", "IllConditionedQuadratic", "Portfolio"] 

for problem in problems
    for ls in [LS_ONLY_SECANT, LS_ADAPTIVE, LS_BACKTRACKING_AND_SECANT, LS_SECANT_WITH_BACKTRACKING, LS_ADAPTIVE_AND_SECANT, LS_ADAPTIVE_ZERO_AND_SECANT, LS_SECANT_3, LS_SECANT_5, LS_SECANT_7, LS_SECANT_12]
        dimensions = if problem in ["OEDP_A", "OEDP_D", "IllConditionedQuadratic"]
            collect(500:500:5000)
        elseif problem == "Portfolio"
            [800, 1200, 1500]
        else
            collect(100:100:1000).^2
        end
        merge_csvs(problem, ls, dimensions=dimensions)
    end
end

