# This example highlights the use of a linear minimization oracle
# using an LP solver defined in MathOptInterface
# we compare the performance of the two LMOs, in- and out of place
#
# to get accurate timings it is important to run twice so that the
# compile time of Julia for the first run is not tainting the results

using FrankWolfe
using ProgressMeter
using Arpack
using Plots
using DoubleFloats
using ReverseDiff

using JSON
using LaTeXStrings
results = JSON.Parser.parsefile("lcg_expensive_data.json")

ref_optimum = results["reference_BCG_primal"]

iteration_list = [
    [x[1] + 1 for x in results["FW"]],
    [x[1] + 1 for x in results["LCG"]],
    [x[1] + 1 for x in results["BLCG"]],
    [x[1] + 1 for x in results["LAFW"]],
    [x[1] + 1 for x in results["BCG"]],
]
time_list = [
    [x[5] for x in results["FW"]],
    [x[5] for x in results["LCG"]],
    [x[5] for x in results["BLCG"]],
    [x[5] for x in results["LAFW"]],
    [x[5] for x in results["BCG"]],
]
primal_gap_list = [
    [x[2] - ref_optimum for x in results["FW"]],
    [x[2] - ref_optimum for x in results["LCG"]],
    [x[2] - ref_optimum for x in results["BLCG"]],
    [x[2] - ref_optimum for x in results["LAFW"]],
    [x[2] - ref_optimum for x in results["BCG"]],
]
dual_gap_list = [
    [x[4] for x in results["FW"]],
    [x[4] for x in results["LCG"]],
    [x[4] for x in results["BLCG"]],
    [x[4] for x in results["LAFW"]],
    [x[4] for x in results["BCG"]],
]
label = [L"\textrm{FW}", L"\textrm{L-CG}", L"\textrm{BL-CG}", L"\textrm{L-AFW}", L"\textrm{BCG}"]

plot_results(
    [primal_gap_list, primal_gap_list, dual_gap_list, dual_gap_list],
    [iteration_list, time_list, iteration_list, time_list],
    label,
    [L"\textrm{Iteration}", L"\textrm{Time}", L"\textrm{Iteration}", L"\textrm{Time}"],
    [L"\textrm{Primal Gap}", L"\textrm{Primal Gap}", L"\textrm{Dual Gap}", L"\textrm{Dual Gap}"],
    xscalelog=[:log, :identity, :log, :identity],
    legend_position=[:bottomleft, nothing, nothing, nothing],
    filename="lcg_expensive.pdf",
)
