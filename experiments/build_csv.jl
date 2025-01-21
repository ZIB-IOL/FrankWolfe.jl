using DataFrames
using CSV

include("utilities.jl")

function build_non_grouped_csv(problem; dimensions=collect(100:100:1000), seeds=collect(1:5))

    function set_up_data(df, dimensions, seeds)
        df[!, :seed] = repeat(seeds, length(dimensions))
        df[!, :dimension] = vcat([fill(i, length(seeds)) for i in dimensions]...)
    end

    df = DataFrame()
    set_up_data(df, dimensions, seeds)

    for ls in [LS_ONLY_SECANT, LS_SECANT_WITH_BACKTRACKING, LS_ADAPTIVE, LS_BACKTRACKING_AND_SECANT, LS_ADAPTIVE_AND_SECANT, LS_ADAPTIVE_ZERO_AND_SECANT, LS_SECANT_3, LS_SECANT_5, LS_SECANT_7, LS_SECANT_12]
       if problem == "Nuclear" && ls == LS_SECANT_WITH_BACKTRACKING
            continue
        end
        df_temp = DataFrame(CSV.File(joinpath(@__DIR__, "csv/" * problem * "/" * string(ls) * ".csv"))) 
        
        df[!, Symbol(string(ls)*"_Time")] = df_temp[!, :time]
        df[!, Symbol(string(ls)*"_Primal")] = df_temp[!, :primal]
        df[!, Symbol(string(ls)*"_DualGap")] = df_temp[!, :dual_gap]
        df[!, Symbol(string(ls)*"_SmallestDualGap")] = df_temp[!, :smallest_dual_gap]
        df[!, Symbol(string(ls)*"_Iterations")] = df_temp[!, :iterations]
        if ls == LS_ADAPTIVE
            df[!, Symbol(string(ls)*"_LineSearchIter")] = df_temp[!, :average_iter]
        elseif is_type_secant(ls)
            df[!, Symbol(string(ls)*"_LineSearchIter")] = df_temp[!, :average_iter] .- 2
        end
    end

    df[!,:minimumTime] = if problem != "Nuclear" 
        min.(
        df[!,Symbol(string(LS_ONLY_SECANT) * "_Time")], 
        df[!,Symbol(string(LS_SECANT_WITH_BACKTRACKING) * "_Time")], 
        df[!,Symbol(string(LS_ADAPTIVE) * "_Time")], 
        df[!,Symbol(string(LS_BACKTRACKING_AND_SECANT) * "_Time")],
        df[!,Symbol(string(LS_ADAPTIVE_AND_SECANT) * "_Time")],
        df[!,Symbol(string(LS_ADAPTIVE_ZERO_AND_SECANT) * "_Time")],
        df[!,Symbol(string(LS_SECANT_3) * "_Time")],
        df[!,Symbol(string(LS_SECANT_5) * "_Time")],
        df[!,Symbol(string(LS_SECANT_7) * "_Time")],
        df[!,Symbol(string(LS_SECANT_12) * "_Time")]
    )
    else
        min.(
        df[!,Symbol(string(LS_ONLY_SECANT) * "_Time")], 
        df[!,Symbol(string(LS_ADAPTIVE) * "_Time")], 
        df[!,Symbol(string(LS_BACKTRACKING_AND_SECANT) * "_Time")],
        df[!,Symbol(string(LS_ADAPTIVE_AND_SECANT) * "_Time")],
        df[!,Symbol(string(LS_ADAPTIVE_ZERO_AND_SECANT) * "_Time")],
        df[!,Symbol(string(LS_SECANT_3) * "_Time")],
        df[!,Symbol(string(LS_SECANT_5) * "_Time")],
        df[!,Symbol(string(LS_SECANT_7) * "_Time")],
        df[!,Symbol(string(LS_SECANT_12) * "_Time")]
    )
    end

    file_name = joinpath(@__DIR__, "csv/" * problem * "_non_grouped.csv")
    CSV.write(file_name, df, append=false)
    println("\n")
end

function build_summary(problem; time_slots=[0, 10, 300, 900, 1800, 2700], dimensions=collect(100:100:1000), by_time=true)
    df = DataFrame()
    df_ng = DataFrame(CSV.File(joinpath(@__DIR__, "csv/" * problem * "_non_grouped.csv")))

    for ls in [LS_ONLY_SECANT, LS_SECANT_WITH_BACKTRACKING, LS_ADAPTIVE, LS_BACKTRACKING_AND_SECANT, LS_ADAPTIVE_AND_SECANT, LS_ADAPTIVE_ZERO_AND_SECANT, LS_SECANT_3, LS_SECANT_5, LS_SECANT_7, LS_SECANT_12]
        if problem == "Nuclear" && ls == LS_SECANT_WITH_BACKTRACKING
            continue
        end
        times = []
        dual_gap_all = []
        dual_gap_all_sd = []
        dual_gap = []
        dual_gap_sd = []
        iterations_all = []
        iterations_solved = []

        filters = by_time ? time_slots : dimensions

        for filter in filters
            instances = by_time ? findall(x -> x>filter, df_ng[!,:minimumTime]) : findall(x -> x==filter, df_ng[!,:dimension])
            not_solved = findall(x-> x > 1e-7, df_ng[instances, Symbol(string(ls)*"_SmallestDualGap")])
            push!(times, geom_shifted_mean(df_ng[instances, Symbol(string(ls)*"_Time")], shift=1.0))
            push!(dual_gap_all_sd, geo_standard_deviation(df_ng[instances, Symbol(string(ls)*"_DualGap")], geom_shifted_mean(df_ng[instances, Symbol(string(ls)*"_DualGap")], shift=1e-5)))
            push!(dual_gap_all, geom_shifted_mean(df_ng[instances, Symbol(string(ls)*"_DualGap")], shift=1e-5))
            push!(dual_gap, geom_shifted_mean(df_ng[intersect(instances,not_solved), Symbol(string(ls)*"_DualGap")], shift=1e-8))
            push!(dual_gap_sd, geo_standard_deviation(df_ng[intersect(instances,not_solved), Symbol(string(ls)*"_DualGap")], geom_shifted_mean(df_ng[intersect(instances,not_solved), Symbol(string(ls)*"_DualGap")], shift=1e-8)))
            push!(iterations_all, custom_mean(df_ng[instances, Symbol(string(ls)*"_Iterations")]))
            push!(iterations_solved, custom_mean(df_ng[setdiff(instances, not_solved), Symbol(string(ls)*"_Iterations")]))
        end

        if by_time
            df[!,:TimeSlots] = time_slots
        else
            df[!,:Dimension] = dimensions
        end

        df[!, Symbol(string(ls)*"_Time")] = times
        df[!, Symbol(string(ls)*"_DualGap")] = dual_gap_all
        df[!, Symbol(string(ls)*"_DualGapSD")] = dual_gap_all_sd
        df[!, Symbol(string(ls)*"_DualGapNotSolved")] = dual_gap
        df[!, Symbol(string(ls)*"_DualGapNotSolvedSD")] = dual_gap_sd
        df[!, Symbol(string(ls)*"_IterationsAll")] = iterations_all
        df[!, Symbol(string(ls)*"_IterationsSolved")] = iterations_solved
    end

    summary_by = by_time ? "difficulty" : "dimension"
    file_name = joinpath(@__DIR__, "csv/" * problem * "_grouped_by_" * summary_by * ".csv")
    CSV.write(file_name, df, append=false)
    #println("\n")
end

problems = ["OEDP_A", "OEDP_D", "Nuclear", "Birkhoff", "QuadraticProbSimplex", "Spectrahedron", "IllConditionedQuadratic"] 

for problem in problems
    @show problem
    dimensions = problem in ["OEDP_A", "OEDP_D", "IllConditionedQuadratic"] ? collect(500:500:5000) : collect(100:100:1000).^2
   build_non_grouped_csv(problem, dimensions=dimensions)
   build_summary(problem, by_time=true) # difficulty
    build_summary(problem, by_time=false, dimensions=dimensions) # dimension
end