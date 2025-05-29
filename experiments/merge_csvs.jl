using CSV
using DataFrames

include("utilities.jl")

function add_failed_instances(seed, dim, ls_variant)
     df = if is_type_secant(ls_variant)
            DataFrame(seed=seed, 
            dimension=dim, 
            time=3600.0, 
            fw_time=3600.0, 
            primal = Inf, 
            dual_gap=Inf, 
            smallest_dual_gap=Inf, 
            iterations=0,
            iter_not_converging =0,
            fallback_helped = 0,
            average_iter = 0,
            std_iter = 0,
            average_gap = Inf,
            std_gap = 0
        )
    elseif ls_variant == LS_ADAPTIVE
            DataFrame(seed=seed, 
            dimension=dim, 
            time=3600.0, 
            fw_time=3600.0, 
            primal = Inf, 
            dual_gap=Inf, 
            smallest_dual_gap=Inf, 
            iterations=0,
            average_iter = 0,
            std_iter = 0
        )
    else
            DataFrame(seed=seed, 
            dimension=dim, 
            time=3600.0, 
            fw_time=3600.0, 
            primal = Inf, 
            dual_gap=Inf, 
            smallest_dual_gap=Inf, 
            iterations=0,
        )
    end
    return df
end

function merge_csvs(problem, ls_variant; dimensions = collect(100:100:1000))
    seeds = collect(1:5)

    start_seed = if (problem == "OEDP_A" && ls_variant == LS_MONOTONIC) || (problem == "OEDP_D" && ls_variant == LS_BACKTRACKING)
        2
    else
        1
    end
    file_name = joinpath(@__DIR__, "csv/" * problem * "/" * string(ls_variant) * "_" * string(dimensions[1]) * "_" * string(seeds[start_seed]) * ".csv")
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
                 if isfile(file_name)
                    df_temp = DataFrame(CSV.File(file_name))
                    append!(df, df_temp)
                else
                    println("Problem: $(problem) Line Search variant: $(string(ls_variant)) Dimension: $(dim) Seed: $(seed)")
                    if problem == "Portfolio" || (problem == "Spectrahedron" && ls_variant in [LS_BACKTRACKING, LS_GOLDEN_RATIO])
                        nothing
                    else
                        df_temp = add_failed_instances(seed, dim, ls_variant)
                        append!(df, df_temp)
                    end
                end
            catch 
                println("Problem: $(problem) Line Search variant: $(string(ls_variant)) Dimension: $(dim) Seed: $(seed)")
                println(e)
            end

        end
    end

    file_name = joinpath(@__DIR__, "csv/" * problem * "/" * string(ls_variant) * ".csv")
    CSV.write(file_name, df, append=false)
end

problems = problems = ["OEDP_A", "OEDP_D", "Nuclear", "Birkhoff", "QuadraticProbSimplex", "Spectrahedron", "IllConditionedQuadratic", "Portfolio"] 
line_searches = [LS_ONLY_SECANT, LS_ADAPTIVE, LS_BACKTRACKING_AND_SECANT, LS_SECANT_WITH_BACKTRACKING, LS_ADAPTIVE_AND_SECANT, LS_ADAPTIVE_ZERO_AND_SECANT, LS_SECANT_3, LS_SECANT_5, LS_SECANT_7, LS_SECANT_12, LS_MONOTONIC, LS_AGNOSTIC, LS_ADAPTIVE_ZERO, LS_BACKTRACKING, LS_GOLDEN_RATIO]
line_searches = [LS_ONLY_SECANT, LS_ADAPTIVE, LS_MONOTONIC, LS_AGNOSTIC, LS_ADAPTIVE_ZERO, LS_BACKTRACKING, LS_GOLDEN_RATIO]
for problem in problems
    for ls in line_searches
        dimensions = get_dimensions(problem)
        merge_csvs(problem, ls, dimensions=dimensions)
        println("\n")
    end
end

