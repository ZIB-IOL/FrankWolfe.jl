# This example highlights the use of a linear minimization oracle
# using an LP solver defined in MathOptInterface
# we compare the performance of the two LMOs, in- and out of place
#
# to get accurate timings it is important to run twice so that the compile time of Julia for the first run
# is not tainting the results

include("activate.jl")

using JSON
using LaTeXStrings
results = JSON.Parser.parsefile("moi_optimizer_results.json")

iteration_list =
    [[x[1] + 1 for x in results["trajectory_lmo"]], [x[1] + 1 for x in results["trajectory_moi"]]]
time_list = [[x[5] for x in results["trajectory_lmo"]], [x[5] for x in results["trajectory_moi"]]]
primal_gap_list =
    [[x[2] for x in results["trajectory_lmo"]], [x[2] for x in results["trajectory_moi"]]]
dual_gap_list =
    [[x[4] for x in results["trajectory_lmo"]], [x[4] for x in results["trajectory_moi"]]]
label =
    [L"\textrm{Closed-form LMO}", L"\textrm{MOI LMO}", L"\textrm{LMO Blas}", L"\textrm{MOI Blas}"]

FrankWolfe.plot_results(
    [primal_gap_list, primal_gap_list, dual_gap_list, dual_gap_list],
    [iteration_list, time_list, iteration_list, time_list],
    label,
    [L"\textrm{Iteration}", L"\textrm{Time}", L"\textrm{Iteration}", L"\textrm{Time}"],
    [L"\textrm{Primal Gap}", L"\textrm{Primal Gap}", L"\textrm{Dual Gap}", L"\textrm{Dual Gap}"],
    xscalelog=[:log, :identity, :log, :identity],
    legend_position=[nothing, :topright, nothing, nothing],
    #filename="results.pdf",
)
