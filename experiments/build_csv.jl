using DataFrames
using CSV
using Printf

include("utilities.jl")

function build_non_grouped_csv(problem; dimensions=collect(100:100:1000), seeds=collect(1:5))

    function set_up_data(df, dimensions, seeds, problem)
        if problem == "Portfolio"
            df[!, :seed] = [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 5]
            df[!, :dimension] = [800, 800, 800, 800, 1200, 1200, 1200, 1200, 1500, 1500, 1500, 1500, 1500]
        else
            df[!, :seed] = repeat(seeds, length(dimensions))
            df[!, :dimension] = vcat([fill(i, length(seeds)) for i in dimensions]...)
        end
    end

    df = DataFrame()
    set_up_data(df, dimensions, seeds, problem)

    minimumTime = fill(Inf, nrow(df))

    line_searches = [LS_ONLY_SECANT, LS_SECANT_WITH_BACKTRACKING, LS_ADAPTIVE, LS_BACKTRACKING_AND_SECANT, LS_ADAPTIVE_AND_SECANT, LS_ADAPTIVE_ZERO_AND_SECANT, LS_SECANT_3, LS_SECANT_5, LS_SECANT_7, LS_SECANT_12, LS_MONOTONIC, LS_AGNOSTIC, LS_ADAPTIVE_ZERO]

    line_searches = [LS_ONLY_SECANT, LS_ADAPTIVE, LS_MONOTONIC, LS_AGNOSTIC, LS_ADAPTIVE_ZERO, LS_GOLDEN_RATIO, LS_BACKTRACKING]
    for ls in line_searches
        if problem == "Spectrahedron" && ls in [LS_GOLDEN_RATIO]
            continue
        end
        df_temp = DataFrame(CSV.File(joinpath(@__DIR__, "csv/" * problem * "/" * string(ls) * ".csv"))) 
        df_temp[df_temp.time .> 3600.00, :time] .= 3600.00
        
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
        minimumTime = min.(minimumTime, df_temp[!, :time])
    end

    df[!,:minimumTime] = minimumTime

    file_name = joinpath(@__DIR__, "csv/" * problem * "_non_grouped.csv")
    CSV.write(file_name, df, append=false)
    println("\n")
end

function build_summary(problem; time_slots=[0, 10, 300, 900, 1800, 2700], dimensions=collect(100:100:1000), by_time=true, table=false)
    df = DataFrame()
    df_ng = DataFrame(CSV.File(joinpath(@__DIR__, "csv/" * problem * "_non_grouped.csv")))

    line_searches = [LS_ONLY_SECANT, LS_ADAPTIVE, LS_MONOTONIC, LS_AGNOSTIC, LS_ADAPTIVE_ZERO, LS_GOLDEN_RATIO, LS_BACKTRACKING]
    line_searches = table ? [LS_ONLY_SECANT, LS_ADAPTIVE, LS_AGNOSTIC, LS_BACKTRACKING] : line_searches

    for ls in line_searches
        if problem == "Nuclear" && ls == LS_SECANT_WITH_BACKTRACKING
            continue
        end
        if problem == "Spectrahedron" && ls in [LS_GOLDEN_RATIO]
            continue
        end
        num_instances = []
        times = []
        dual_gap_all = []
        dual_gap_all_sd = []
        dual_gap = []
        dual_gap_sd = []
        iterations_all = []
        iterations_solved = []

        filters = by_time ? time_slots : dimensions
        not_solved_all = findall(x-> x > 1e-7, df_ng[!, Symbol(string(ls)*"_SmallestDualGap")])
        for filter in filters
            instances = by_time ? findall(x -> x>filter, df_ng[!,:minimumTime]) : findall(x -> x==filter, df_ng[!,:dimension])
            push!(num_instances, length(instances))
            not_solved = intersect(instances, not_solved_all)
            @show filter, length(not_solved)
            push!(times, geom_shifted_mean(df_ng[instances, Symbol(string(ls)*"_Time")], shift=1.0))
            push!(dual_gap_all_sd, geo_standard_deviation(df_ng[instances, Symbol(string(ls)*"_DualGap")], geom_shifted_mean(df_ng[instances, Symbol(string(ls)*"_DualGap")], shift=1e-5)))
            push!(dual_gap_all, geom_shifted_mean(df_ng[instances, Symbol(string(ls)*"_DualGap")], shift=1))
            push!(dual_gap, geom_shifted_mean(df_ng[not_solved, Symbol(string(ls)*"_DualGap")], shift=1))
            push!(dual_gap_sd, geo_standard_deviation(df_ng[not_solved, Symbol(string(ls)*"_DualGap")], geom_shifted_mean(df_ng[not_solved, Symbol(string(ls)*"_DualGap")], shift=1e-8)))
            push!(iterations_all, custom_mean(df_ng[instances, Symbol(string(ls)*"_Iterations")]))
            push!(iterations_solved, custom_mean(df_ng[setdiff(instances, not_solved), Symbol(string(ls)*"_Iterations")]))
        end

        if by_time
            df[!,:TimeSlots] = time_slots
        else
            df[!,:Dimension] = dimensions
        end

        df[!,:NumInstances] = num_instances

        # rounding
        non_inf = findall(isfinite, times)
        times[non_inf] = round.(times[non_inf], digits=1)

        non_inf = findall(isfinite, iterations_all)
        iterations_all[non_inf] = convert.(Int64, round.(iterations_all[non_inf]))

        non_inf = findall(isfinite, iterations_solved)
        iterations_solved[non_inf] = convert.(Int64, round.(iterations_solved[non_inf]))

        iter_time_ratio = round.(iterations_all ./ times, digits=1)

        if table
            df[!, Symbol(string(ls)*"_Time")] = times
            dual_gap_print = []
            for d in dual_gap
                if d == Inf
                    push!(dual_gap_print, "<1e-7")
                elseif d == 0.0
                    push!(dual_gap_print, "[<1e-7]")
                else
                    push!(dual_gap_print, @sprintf("%.2e", d))
                end
            end
            df[!, Symbol(string(ls)*"_DualGapNotSolved")] = dual_gap_print
            iteration_print = []
            for x in iterations_solved
                if x == Inf
                    push!(iteration_print, "--")
                elseif x > 10e5
                    push!(iteration_print, "> $(Int(floor(x/10e5)))M")
                else
                    push!(iteration_print, string(x))
                end
            end
            df[!, Symbol(string(ls)*"_IterationsSolved")] = iteration_print
            df[!,Symbol(string(ls)*"_IterTimeRatio")] = iter_time_ratio
        else
            df[!, Symbol(string(ls)*"_Time")] = times
            df[!, Symbol(string(ls)*"_DualGap")] = dual_gap_all
            df[!, Symbol(string(ls)*"_DualGapSD")] = dual_gap_all_sd
            df[!, Symbol(string(ls)*"_DualGapNotSolved")] = dual_gap
            df[!, Symbol(string(ls)*"_DualGapNotSolvedSD")] = dual_gap_sd
            df[!, Symbol(string(ls)*"_IterationsAll")] = iterations_all
            df[!, Symbol(string(ls)*"_IterationsSolved")] = iterations_solved
        end

    end

    summary_by = by_time ? "difficulty" : "dimension"
    summary_by = table ? summary_by * "_table" : summary_by
    file_name = joinpath(@__DIR__, "csv/" * problem * "_grouped_by_" * summary_by * ".csv")
    CSV.write(file_name, df, append=false)
    #println("\n")
end

function summary_table()
    problems = ["Birkhoff", "IllConditionedQuadratic", "Nuclear", "OEDP_A", "OEDP_D", "Portfolio", "QuadraticProbSimplex", "Spectrahedron"]
    df_summary_table = DataFrame([[],[],[], [], [], [], [], [], [], [], [], [], [], []], ["Problem", "Instances", "TimeS", "DualGapS", "IterationS", "TimeAD", "DualGapAD", "IterationAD", "TimeAG", "DualGapAG", "IterationAG", "TimeB", "DualGapB", "IterationsB"])

    for problem in problems
        file_name = joinpath(@__DIR__, "csv/" * problem * "_grouped_by_difficulty_table.csv")
        df = DataFrame(CSV.File(file_name))
@show df[1,:NumInstances]
        push!(df_summary_table, [problem, df[1,:NumInstances],df[1,:LS_ONLY_SECANT_Time],df[1,:LS_ONLY_SECANT_DualGapNotSolved],df[1, :LS_ONLY_SECANT_IterationsSolved],df[1,:LS_ADAPTIVE_Time],df[1,:LS_ADAPTIVE_DualGapNotSolved],df[1,:LS_ADAPTIVE_IterationsSolved],df[1,:LS_AGNOSTIC_Time],df[1,:LS_AGNOSTIC_DualGapNotSolved],df[1,:LS_AGNOSTIC_IterationsSolved], df[1,:LS_BACKTRACKING_Time], df[1,:LS_BACKTRACKING_DualGapNotSolved], df[1,:LS_BACKTRACKING_IterationsSolved]])
    end

    file_name = joinpath(@__DIR__, "csv/summary_table.csv")
    CSV.write(file_name, df_summary_table, append=false)
end

problems = ["OEDP_A", "OEDP_D", "Nuclear", "Birkhoff", "QuadraticProbSimplex", "IllConditionedQuadratic", "Spectrahedron", "Portfolio"] #"Spectrahedron",
for problem in problems
    @show problem
    dimensions = get_dimensions(problem)
    build_non_grouped_csv(problem, dimensions=dimensions)
    build_summary(problem, by_time=true) # difficulty
    build_summary(problem, by_time=false, dimensions=dimensions) # dimension
    build_summary(problem, by_time=true, table=true) # difficulty
    build_summary(problem, by_time=false, dimensions=dimensions, table=true) # dimension
end

summary_table()
