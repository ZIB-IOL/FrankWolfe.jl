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
        for i in range(1,step=iter_skip,stop=length(data))
            println(io, join(data[i], " "))
        end
    end
end


function extract_data(problem, ls; subfolder="", termination=false, trajectory=false, dim=0, seed=0)
    data = []
    if trajectory
        @assert dim > 0 && seed > 0
        df = DataFrame(CSV.File(joinpath(@__DIR__, "../csv/" * problem * "/trajectory/" * string(ls) * "_" * string(dim) * "_" * string(seed) * ".csv")))
        for row in df
            push!(data, collect(row))
        end
    elseif termination
        df = DataFrame(CSV.File(joinpath(@__DIR__, "../csv/" * problem * "_non_grouped.csv")))
        termination = [row > 1e-7 ? 0 : 1 for row in df[!,Symbol(string(ls) * "_SmallestDualGap")]]
        df[df[!, Symbol(string(ls)*"_Time")].>3600, Symbol(string(ls)*"_Time")] .= 3600
        df[!,:boolTerm] = termination

        filter!(row -> !(row.boolTerm == 0),  df)
        x = sort(df[!,Symbol(string(ls)*"_Time")])
        for i in 1:nrow(df)
            push!(data, [x[i], i])
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
for ls in linesearches
    for problem in problems
        if problem == "Nuclear" && ls == LS_SECANT_WITH_BACKTRACKING
            continue
        end
        data = extract_data(problem, ls, termination=true)
        export_data(data, ["time", "termination"], filename_prefix=problem * "_" * string(ls), filename_suffix="termination", compute_FWgaps=false)

        data = extract_data(problem, ls)
        export_data(data, ["time", "dual_gap", "dual_gap_sd", "dual_gap_ns", "dual_gap_ns_sd", "iterations", "iterations_s"], filename_prefix=problem * "_" * string(ls), filename_suffix="dual_gap", compute_FWgaps=false)
    end
end

# data_trajectories
#data = extract_data(problem, ls, trajectory=true, dim=100, seed=0)
#export_data(data, ["iteration", "primal", "dual_bound", "dual_gap", "time"], filename_suffix="trajectory", compute_FWgaps=false)

