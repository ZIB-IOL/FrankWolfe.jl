using CSV
using DataFrames
using Plots

include("../utilities.jl")

"""
Outputs data in a latex-readable format.
Original code by Zev Woodstock here: https://github.com/zevwoodstock/BlockFW/blob/main/plot_utils.jl
"""
function export_data(
    data,
    labels;
    filename_prefix="Boscia",
    filename_suffix="",
    iter_skip=1,
    compute_FWgaps=true,
)
    file_name = joinpath(@__DIR__, "data/" * filename_prefix * "_" * filename_suffix * ".txt")
    open(file_name, "w") do io 
        println(io, join(labels, " "))
        stop = length(data) >= 2000 ? 2000 : length(data)
        for i in range(1,step=iter_skip,stop=stop)
            println(io, join(data[i], " "))
        end
    end
end

# Function to plot two lines and save the plot to a file
function plot_lines_save(x1, y1, x2, y2; label1="Line 1", label2="Line 2", title="Plot", xlabel="x", ylabel="y")
    # Create the plot
    plot(x1, y1, label=label1, xlabel=xlabel, ylabel=ylabel, title=title, color=:blue)
    plot!(x2, y2, label=label2, color=:red)  # Add second line to the same plot
    
    # Save the plot to a file
    filename = joinpath(@__DIR__, "figures/" * title * ".pdf")
    savefig(filename)
end

function plot_subplots_save(x1, y1, x2, y2, x3, y3, x4, y4; label1="Line 1", label2="Line 2", title="Plot 1", x1label="", y1label="", x2label="", y2label="")
    # Create a 2x2 grid of subplots
    #x1scale = x1[end] > 1000 && x3[end] > 1000 ? :log10 : :identity
    x1scale = :identity
    #x2scale = x2[end] > 300 && x4[end] > 300 ? :log10 : :identity
    x2scale = :identity
    p1 = plot(x1, y1, label=label1, xlabel=x1label, ylabel=y1label, color=:blue,xscale=x1scale) #xscale=:log10
    plot!(x3, y3, label=label2, color=:red)  # Adding second line to the first subplot

    p2 = plot(x2, y1, label=label1, xlabel=x2label, ylabel=y1label, color=:blue, xscale=x2scale)
    plot!(x4, y3, label=label2, color=:red)  # Adding second line to the second subplot

    p3 = plot(x1, y2, label=label1, xlabel=x1label, ylabel=y2label, color=:blue, xscale=x1scale)
    plot!(x3, y4, label=label2, color=:red)  # Adding second line to the third subplot

    p4 = plot(x2, y2, label=label1, xlabel=x2label, ylabel=y2label, color=:blue, xscale=x2scale)
    plot!(x4, y4, label=label2, color=:red)  # Adding second line to the fourth subplot

    # Combine the plots into one figure with a 2x2 layout
    plot(p1, p2, p3, p4, layout=(2, 2))

    # Save the plot to a file
    filename = joinpath(@__DIR__, "figures/" * title * ".pdf")
    savefig(filename)
end


function extract_data(problem, ls; subfolder="", termination=false, trajectory=false, termination_iter=false, no_termination_iter=false, secant_dual_gap=false, secant_iter=false, dim=0, seed=0, vanilla=false)
    data = []
    if trajectory
        @assert dim > 0 && seed > 0
        traj_file = if vanilla 
            joinpath(@__DIR__, "../csv/Vanilla/" * problem * "/trajectory/" * string(ls) * "_" * string(dim) * "_" * string(seed) * ".csv")
        else
            joinpath(@__DIR__, "../csv/" * problem * "/trajectory/" * string(ls) * "_" * string(dim) * "_" * string(seed) * ".csv")
        end
        if !isfile(traj_file)
            println("Trajectory of $(problem)_$(string(ls)) with dimension $(dim) and seed $(seed) does not exists.")
            return nothing
        end
        df = DataFrame(CSV.File(traj_file))
        df[1,:time] = 0.0001
        length = nrow(df)
        indices =  if length > 1000 
            vcat(collect(1:999), Int.(round.(collect(1000:(length-1000)/1000:(length-1000))))) 
        else
             Int.(collect(1:length))
        end
        for row in eachrow(df)
            if rownumber(row) in indices
                push!(data, collect(row))
            end
        end
    elseif secant_iter
        df = DataFrame(CSV.File(joinpath(@__DIR__, "../csv/" * problem * "_non_grouped.csv")))
        x = sort(df[!,Symbol(string(ls)*"_LineSearchIter")])
        for i in 1:nrow(df)
            push!(data, [x[i], i])
        end
    elseif termination
        df = DataFrame(CSV.File(joinpath(@__DIR__, "../csv/" * problem * "_non_grouped.csv")))
        termination = [row > 1e-5 ? 0 : 1 for row in df[!,Symbol(string(ls) * "_SmallestDualGap")]]
        df[!,:boolTerm] = termination
        if termination_iter
            filter!(row -> !(row.boolTerm == 0),  df)
            x = sort(df[!,Symbol(string(ls)*"_LineSearchIter")])
            for i in 1:nrow(df)
                push!(data, [x[i], i])
            end
        elseif no_termination_iter
            filter!(row -> !(row.boolTerm == 1),  df)
            x = sort(df[!,Symbol(string(ls)*"_LineSearchIter")])
            for i in 1:nrow(df)
                push!(data, [x[i], i])
            end
        else
            df[df[!, Symbol(string(ls)*"_Time")].>3600, Symbol(string(ls)*"_Time")] .= 3600
            filter!(row -> !(row.boolTerm == 0),  df)
            x = sort(df[!,Symbol(string(ls)*"_Time")])
            for i in 1:nrow(df)
                push!(data, [x[i], i])
            end
        end
    elseif secant_dual_gap
        df = DataFrame(CSV.File(joinpath(@__DIR__, "../csv/" * problem * "_non_grouped.csv")))
        idx = sortperm(df[!,Symbol(string(ls)*"_LineSearchIter")])
        for i in idx
            push!(data, [df[i,Symbol(string(ls)*"_LineSearchIter")], df[i, Symbol(string(ls)*"_DualGap")]])
        end
    else
        df = DataFrame(CSV.File(joinpath(@__DIR__, "../csv/" * problem * "_grouped_by_dimension.csv")))
        for row in eachrow(df)
            push!(data, collect(row))
        end
    end

    return data
end

linesearches = [LS_ADAPTIVE, LS_ONLY_SECANT]
problems = ["Birkhoff", "IllConditionedQuadratic", "Nuclear", "OEDP_A", "OEDP_D", "QuadraticProbSimplex", "Spectrahedron", "Portfolio"] #"Portfolio"
#=
println("Termination and Dual Gap spread plots")
for ls in linesearches
    for problem in problems
        if problem == "Nuclear" && ls == LS_SECANT_WITH_BACKTRACKING
            continue
        end
       # data = extract_data(problem, ls, termination=true)
       # export_data(data, ["time", "termination"], filename_prefix=problem * "_" * string(ls), filename_suffix="termination", compute_FWgaps=false)

        if is_type_secant(ls) || ls == LS_ADAPTIVE
            data = extract_data(problem, ls, termination=true,termination_iter=true)
            export_data(data, ["ls_iter", "termination"], filename_prefix=problem * "_" * string(ls), filename_suffix="termination_iter", compute_FWgaps=false)
        end

        if ls == LS_ONLY_SECANT
            data = extract_data(problem, ls, secant_dual_gap=true)
            export_data(data, ["ls_iter", "dual_gap"], filename_prefix=problem * "_" * string(ls),filename_suffix="iter_dual_gap", compute_FWgaps=false)

            data = extract_data(problem, ls, secant_iter=true)
            export_data(data, ["ls_iter", "no_instance"], filename_prefix=problem * "_" * string(ls), filename_suffix="secant_iter", compute_FWgaps=false)
        end

        #if problem in ["OEDP_A", "OEDP_D", "Portfolio"] && is_type_secant(ls)
        #    data = extract_data(problem, ls, termination=true,no_termination_iter=true)
        #    export_data(data, ["ls_iter", "termination"], filename_prefix=problem * "_" * string(ls), filename_suffix="no_termination_iter", compute_FWgaps=false)
        #end

        data = extract_data(problem, ls)
        export_data(data, ["dimension", "time", "dual_gap", "dual_gap_sd", "dual_gap_ns", "dual_gap_ns_sd", "iterations", "iterations_s"], filename_prefix=problem * "_" * string(ls), filename_suffix="dual_gap", compute_FWgaps=false)
    end
end
=#
println("Trajectory Plots")
linesearches = [LS_ADAPTIVE, LS_ADAPTIVE_ZERO, LS_AGNOSTIC, LS_GOLDEN_RATIO, LS_BACKTRACKING, LS_MONOTONIC, LS_ONLY_SECANT]
for ls in linesearches
    data = extract_data("Spectrahedron", ls, trajectory=true, dim=90000, seed=1, vanilla=true)
    export_data(data, ["iteration", "primal", "dual_bound", "dual_gap", "time", "step_size"],filename_prefix="trajectory/" * "Spectrahedron_Vanilla" * "_" * string(90000) * "_" * string(1) * "_" * string(ls), filename_suffix="trajectory", compute_FWgaps=false)

    data = extract_data("Nuclear", ls, trajectory=true, dim=10000, seed=1, vanilla=true)
    export_data(data, ["iteration", "primal", "dual_bound", "dual_gap", "time", "step_size"],filename_prefix="trajectory/" * "Nuclear_Vanilla" * "_" * string(10000) * "_" * string(1) * "_" * string(ls), filename_suffix="trajectory", compute_FWgaps=false)
end

#=
problems = ["Birkhoff", "IllConditionedQuadratic", "Nuclear", "OEDP_A", "OEDP_D", "QuadraticProbSimplex", "Spectrahedron", "Portfolio"]
dimension = [40000, 2000, 10000, 500, 1000, 10000, 90000, 800]
seeds = [1,3,1,5,4,1,1,3]

for ls in linesearches
for i in 1:length(problems)
    @show problems[i]
    @show ls
    data = extract_data(problems[i], ls, trajectory=true, dim=dimension[i], seed=seeds[i])
    export_data(data, ["iteration", "primal", "dual_bound", "dual_gap", "time", "step_size"],filename_prefix="trajectory/" * problems[i] * "_" * string(dimension[i]) * "_" * string(seeds[i]) * "_" * string(ls), filename_suffix="trajectory", compute_FWgaps=false)
end
end
=#
#=
seeds = collect(1:5)
for ls in linesearches
    for problem in problems
        dimensions = get_dimensions(problem)
        @show problem
        for dim in dimensions
            for seed in seeds
                # data_trajectories
                data = extract_data(problem, ls, trajectory=true, dim=dim, seed=seed)
                if data === nothing
                    continue
                end
                export_data(data, ["iteration", "primal", "dual_bound", "dual_gap", "time", "step_size"],filename_prefix="trajectory/" * problem * "_" * string(dim) * "_" * string(seed) * "_" * string(ls), filename_suffix="trajectory", compute_FWgaps=false)
            end 
        end
    end
end
=#

#=
# plots 
println("Plots")
linesearches = [LS_ADAPTIVE, LS_ONLY_SECANT]
problems = ["Birkhoff", "IllConditionedQuadratic", "Nuclear", "OEDP_A", "OEDP_D", "QuadraticProbSimplex", "Spectrahedron", "Portfolio"]
seeds = collect(1:5)
for problem in problems
    dimensions = get_dimensions(problem)
    @show problem
    for dim in dimensions
        for seed in seeds
            # data_trajectories
            data_a = extract_data(problem, LS_ADAPTIVE, trajectory=true, dim=dim, seed=seed)
            data_s = extract_data(problem, LS_ONLY_SECANT, trajectory=true, dim=dim, seed=seed)
            if data_s === nothing || data_a === nothing
                continue
            end
           # @show getindex.(data_a, 1)
           # plot_lines_save(getindex.(data_a, 1), getindex.(data_a, 2), getindex.(data_s, 1), getindex.(data_s, 2), label1="Adaptive", label2="Secant", title=problem * "_" * string(dim) * "_" * string(seed), xlabel="Iteration", ylabel="Primal")
           try
           plot_subplots_save(getindex.(data_a, 1), getindex.(data_a, 2), getindex.(data_a, 5), getindex.(data_a, 4), getindex.(data_s, 1), getindex.(data_s, 2), getindex.(data_s, 5), getindex.(data_s, 4), label1="Adaptive", label2="Secant", title=problem * "_" * string(dim) * "_" * string(seed), x1label="Iteration", y1label="Primal", x2label="Time", y2label="FW Gap")
           catch e
            println("Problem $(problem) with dimension $(dim) and seed $(seed) could not be plotted.")
                println(e)
           end
        end 
    end
end
=#
