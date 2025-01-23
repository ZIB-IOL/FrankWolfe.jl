using CSV
using DataFrames

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


function extract_data(problem, ls; subfolder="", termination=false, trajectory=false, termination_iter=false, dim=0, seed=0)
    data = []
    if trajectory
        @assert dim > 0 && seed > 0
        traj_file = joinpath(@__DIR__, "../csv/" * problem * "/trajectory/" * string(ls) * "_" * string(dim) * "_" * string(seed) * ".csv")
        if !isfile(traj_file)
            println("Trajectory of $(problem)_$(string(ls)) with dimension $(dim) and seed $(seed) does not exists.")
            return nothing
        end
        df = DataFrame(CSV.File(traj_file))
        for row in eachrow(df)
            push!(data, collect(row))
        end
    elseif termination
        df = DataFrame(CSV.File(joinpath(@__DIR__, "../csv/" * problem * "_non_grouped.csv")))
        termination = [row > 1e-7 ? 0 : 1 for row in df[!,Symbol(string(ls) * "_SmallestDualGap")]]
        df[!,:boolTerm] = termination
        if termination_iter
            filter!(row -> !(row.boolTerm == 0),  df)
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
    else
        df = DataFrame(CSV.File(joinpath(@__DIR__, "../csv/" * problem * "_grouped_by_dimension.csv")))
        for row in eachrow(df)
            push!(data, collect(row))
        end
    end

    return data
end

linesearches = [LS_ADAPTIVE, LS_ADAPTIVE_AND_SECANT, LS_BACKTRACKING_AND_SECANT, LS_ONLY_SECANT, LS_SECANT_WITH_BACKTRACKING, LS_SECANT_12, LS_SECANT_3, LS_SECANT_5, LS_SECANT_7, LS_ADAPTIVE_ZERO_AND_SECANT]
problems = ["Birkhoff", "IllConditionedQuadratic", "Nuclear", "OEDP_A", "OEDP_D", "QuadraticProbSimplex", "Spectrahedron"] #"Portfolio"

println("Termination and Dual Gap spread plots")
for ls in linesearches
    for problem in problems
        if problem == "Nuclear" && ls == LS_SECANT_WITH_BACKTRACKING
            continue
        end
        data = extract_data(problem, ls, termination=true)
        export_data(data, ["time", "termination"], filename_prefix=problem * "_" * string(ls), filename_suffix="termination", compute_FWgaps=false)

        if is_type_secant(ls) || ls == LS_ADAPTIVE
            data = extract_data(problem, ls, termination=true,termination_iter=true)
            export_data(data, ["ls_iter", "termination"], filename_prefix=problem * "_" * string(ls), filename_suffix="termination_iter", compute_FWgaps=false)
        end

        data = extract_data(problem, ls)
        export_data(data, ["dimension", "time", "dual_gap", "dual_gap_sd", "dual_gap_ns", "dual_gap_ns_sd", "iterations", "iterations_s"], filename_prefix=problem * "_" * string(ls), filename_suffix="dual_gap", compute_FWgaps=false)
    end
end

println("Trajectory Plots")
problems = ["Birkhoff", "IllConditionedQuadratic", "Nuclear", "OEDP_A", "OEDP_D", "QuadraticProbSimplex", "Spectrahedron", "Portfolio"]
seeds = collect(1:5)
for ls in linesearches
    for problem in problems
        dimensions = if problem in ["IllConditionedQuadratic", "OEDP_A", "OEDP_D"]
            collect(500:500:2000)
        elseif problem == "Portfolio"
            [800, 1200, 1500]
        else
            collect(100:100:300).^2
        end
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



