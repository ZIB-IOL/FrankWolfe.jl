using Random
using FrankWolfe

include("solve_problems.jl")

"""
DON'T FORGET TO ADD e AGAIN ONCE YOU ARE DONE DEBUGGING!!
"""
problem = ENV["PROBLEM"]
seed = parse(Int, ENV["SEED"])
m = parse(Int, ENV["DIMENSION"])
fw_variant = ENV["VARIANT"]

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
elseif ls == "AdaptiveSecant"
    LS_ADAPTIVE_AND_SECANT
elseif ls == "AdaptiveZeroSecant"
    LS_ADAPTIVE_ZERO_AND_SECANT
elseif ls == "Secant_3"
    LS_SECANT_3
elseif ls == "Secant_5"
    LS_SECANT_5
elseif ls == "Secant_7"
    LS_SECANT_7
elseif ls == "Secant_12"
    LS_SECANT_12
elseif ls == "Monotonic"
    LS_MONOTONIC
elseif ls == "Agnostic"
    LS_AGNOSTIC
elseif ls == "AdaptiveZero"
    LS_ADAPTIVE_ZERO
elseif ls == "Backtracking"
    LS_BACKTRACKING
elseif ls == "Goldenratio"
    LS_GOLDEN_RATIO
else
    error("Unknown linesearch")
end

@show problem, string(line_search), m, seed

for seed in seeds
    @show seed
    try
        solve_problems(seed, m, problem, line_search, FW_variant=fw_variant)
    catch e
        showerror(stdout, e, catch_backtrace())
        error_file = problem * "_" * string(line_search) * "_" * fw_variant * ".txt" 
        open(error_file,"a") do io
            println(io, seed, " ", m, " : ", e)
        end
    end
end