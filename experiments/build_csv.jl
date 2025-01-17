using DataFrames
using CSV

inlcude("utilities.jl")

function build_non_grouped_csv(problem; dimensions=collect(100:100:1000), seeds=collect(1:5))

    function set_up_data(df, dimensions, seeds)
        df[!, :seed] = repeat(seeds, length(dimensions))
        df[!, :dimension] = vcat([fill(i^2, length(seeds) for i in dimensions)]...)
    end

    df = DataFrame()
    set_up_data(df, dimensions, seeds)

    for ls in [LS_ONLY_SECANT, LS_SECANT_WITH_BACKTRACKING, LS_ADAPTIVE, LS_BACKTRACKING_AND_SECANT]
        df_temp = DataFrame(CSV.File(joinpath(@__DIR__, "csv/" * problem * "/" * string(ls) * ".csv"))) 
        
        df[!, Symbol(string(ls)*"_Time")] = df_temp[!, :time]
        df[!, Symbol(string(ls)*"_Primal")] = df_temp[!, :primal]
        df[!, Symbol(string(ls)*"_DualGap")] = df_temp[!, :dual_gap]
        df[!, Symbol(sting(ls)*"_SmallestDualGap")] = df_temp[!, :smallest_dual_gap]
        df[!, Symbol(string(ls)*"_Iterations")] = df_temp[!, :iterations]
    end

    df[!,:minimumTime] = min.(
        df[!,Symbol(string(LS_ONLY_SECANT) * "_Time")], 
        df[!,Symbol(string(LS_SECANT_WITH_BACKTRACKING) * "_Time")], 
        df[!,Symbol(string(LS_ADAPTIVE) * "_Time")], 
        df[!,Symbol(string(LS_BACKTRACKING_AND_SECANT) * "_Time")]
    )

    file_name = joinpath(@__DIR__, "csv/" * problem * "_non_grouped.csv")
    CSV.write(file_name, df, append=false)
    println("\n")
end


function build_summary(problem; time_slots=[0, 10, 300, 900, 1800, 2700], dimensions=collect(100:100:1000), by_time=true)
    df = DataFrame()
    df_ng = DataFrame(CSV.File(joinpath(@__DIR__, "csv/" * problem * "_non_grouped.csv")))

    for ls in [LS_ONLY_SECANT, LS_SECANT_WITH_BACKTRACKING, LS_ADAPTIVE, LS_BACKTRACKING_AND_SECANT]
        times = []
        dual_gap_all = []
        dual_gap_all_sd = []
        dual_gap = []
        dual_gap_sd = []
        iterations = []

        for time_slot in time_slots
            instances = by_time ? findall(x -> x>time_slot, df_ng[!,:minimumTime]) : findall(x -> x==dimension^2, df_ng[!,:dimension])
            not_solved = findall(x-> x > 1e-7, df_ng[instances, Symbol(string(ls)*"_SmallestDualGap")])
            push!(times, geom_shifted_mean(df_ng[instances, Symbol(string(ls)*"_Time")], shift=Big"1.0"))
            push!(dual_gap_all_sd, geo_standard_deviation(df_ng[instances, Symbol(string(ls)*"_DualGap")], geom_shifted_mean(df_ng[instances, Symbol(string(ls)*"_DualGap")], shift=1e-8)))
            push!(dual_gap_all, geom_shifted_mean(df_ng[instances, Symbol(string(ls)*"_DualGap")], shift=1e-8))
            push!(dual_gap, geom_shifted_mean(df_ng[intersect(instances,not_solved), Symbol(string(ls)*"_DualGap")], shift=1e-8))
            push!(dual_gap_sd, geo_standard_deviation(df_ng[intersect(instances,not_solved), Symbol(string(ls)*"_DualGap")], geom_shifted_mean(df_ng[intersect(instances,not_solved), Symbol(string(ls)*"_DualGap")], shift=1e-8)))
            push!(iterations, custom_mean(df_ng[instances, Symbol(string(ls)*"_Iterations")]))
        end

        df[!, Symbol(string(ls)*"_Time")] = times
        df[!, Symbol(string(ls)*"_DualGap")] = dual_gap_all
        df[!, Symbol(string(ls)*"_DualGapSD")] = dual_gap_all_sd
        df[!, Symbol(string(ls)*"_DualGapNotSolved")] = dual_gap
        df[!, Symbol(string(ls)*"_DualGapNotSolvedSD")] = dual_gap_sd
        df[!, Symbol(string(ls)*"_Iterations")] = iterations
    end

    summary_by = by_time ? "difficulty" : "dimension"
    file_name = joinpath(@__DIR__, "csv/" * problem * "_grouped_by_" * summary_by * ".csv")
    CSV.write(file_name, df, append=false)
    println("\n")
end

problems = problems = ["OEDP_A", "OEDP_D", "Nuclear", "Birkhoff", "QuadraticProbSimplex", "Spectrahedron", "IllConditionedQuadratic"] 

for problem in problems
   build_non_grouped_csv(problem)
   build_summary(problem, by_time=true) # difficulty
    build_summary(problem, by_time=false) # dimension
end