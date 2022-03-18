# This example highlights the use of a linear minimization oracle
# using an LP solver defined in MathOptInterface
# we compare the performance of the two LMOs, in- and out of place
#
# to get accurate timings it is important to run twice so that the compile time of Julia for the first run
# is not tainting the results

using FrankWolfe
using ProgressMeter
using Arpack
using Plots
using DoubleFloats
using ReverseDiff

using JSON
using LaTeXStrings
results = JSON.Parser.parsefile("movielens_result.json")

ref_optimum = results["trajectory_arr_lazy_ref"][end][2]

iteration_list = [
    [x[1] + 1 for x in results["trajectory_arr_fw"]],
    [x[1] + 1 for x in results["trajectory_arr_lazy"]],
    collect(1:1:length(results["function_values_gd"])),
]
time_list = [
    [x[5] for x in results["trajectory_arr_fw"]],
    [x[5] for x in results["trajectory_arr_lazy"]],
    results["timing_values_gd"],
]
primal_gap_list = [
    [x[2] - ref_optimum for x in results["trajectory_arr_fw"]],
    [x[2] - ref_optimum for x in results["trajectory_arr_lazy"]],
    [x - ref_optimum for x in results["function_values_gd"]],
]
test_list =
    [results["fw_test_values"], results["lazy_test_values"], results["function_values_test_gd"]]

label = [L"\textrm{FW}", L"\textrm{L-CG}", L"\textrm{GD}"]

plot_results(
    [primal_gap_list, primal_gap_list, test_list, test_list],
    [iteration_list, time_list, iteration_list, time_list],
    label,
    [L"\textrm{Iteration}", L"\textrm{Time}", L"\textrm{Iteration}", L"\textrm{Time}"],
    [
        L"\textrm{Primal Gap}",
        L"\textrm{Primal Gap}",
        L"\textrm{Test Error}",
        L"\textrm{Test Error}",
    ],
    xscalelog=[:log, :identity, :log, :identity],
    legend_position=[:bottomleft, nothing, nothing, nothing],
    filename="movielens_result.pdf",
)
