using Random
using FrankWolfe

include("solve_problems.jl")

"""
DON'T FORGET TO ADD e AGAIN ONCE YOU ARE DONE DEBUGGING!!
"""
problem = ENV["PROBLEM"]
seed = parse(Int, ENV["SEED"])
m = parse(Int, ENV["DIMENSION"])

time_limit = 3600 # one hour time limit
seeds = seed == 0 ? [1,2,3,4,5] : [seed]

ls = ENV["LINESEARCH"]
line_search = if ls == "Secant"
    LS_ONLY_SECANT
elseif ls == "SecantBT"
    LS_SECANT_WITH_BACKTRACKING
elseif ls == "Adaptive"
    LS_ADAPTIVE
elseif ls == "BacktrackingSecant"
    LS_BACKTRACKING_AND_SECANT
else
    error("Unknown linesearch")
end

@show problem, string(line_search), m, seed

for seed in seeds
    @show seed
    try
        solve_problems(seed, m, problem, line_search)
    catch e
        showerror(stdout, e, catch_backtrace())
        error_file = problem * "_" * string(line_search) * ".txt" 
        open(error_file,"a") do io
            println(io, seed, " ", m, " : ", e)
        end
    end
end