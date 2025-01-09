using DataFrames
using CSV

inlcude("solve_problems.jl")

function build_non_grouped_csv(problem; dimensions=collect(100:100:1000), seeds=collect(1:5))

    function set_up_data(df, dimensions, seeds)
        df[!, :seed] = repeat(seeds, length(dimensions))
        df[!, :dimension] = vcat([fill(i, length(seeds) for i in dimensions)]...)
    end

    function read_data()
    end

    df = DataFrame()
    set_up_data(df, dimensions, seeds)

    for ls in [LS_ONLY_SECANT, LS_SECANT_WITH_BACKTRACKING, LS_ADAPTIVE, LS_BACKTRACKING_AND_SECANT]
        df_temp = DataFrame(CSV.File(joinpath(@__DIR__, "csv/" * problem * "/" * string(ls) * ".csv"))) 
        
        df[!, Symbol(string(ls)*"_Time")] = df_temp[!, :time]
        df[!, Symbol(string(ls)*"_Primal")] = df_temp[!, :primal]
        df[!, Symbol(string(ls)*"_DualGap")] = df_temp[!, :dual_gap]
        df[!, Symbol(string(ls)*"_Iterations")] = df_temp[!, :iterations]
    end

    file_name = joinpath(@__DIR__, "csv/" * problem * "_non_grouped.csv")
    CSV.write(file_name, df, append=false)
    println("\n")
end

function build_summary_by_difficulty()
end

function build_by_difficulty()
end