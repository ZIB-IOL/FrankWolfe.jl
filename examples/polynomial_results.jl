
using FrankWolfe
using ProgressMeter
using Arpack
using Plots
using DoubleFloats
using ReverseDiff

using JSON
using LaTeXStrings

results = JSON.Parser.parsefile(joinpath(@__DIR__, "polynomial_result.json"))

iteration_list = [
    [x[1] + 1 for x in results["trajectory_arr_lafw"]],
    [x[1] + 1 for x in results["trajectory_arr_bcg"]],
    collect(eachindex(results["function_values_gd"])),
]
time_list = [
    [x[5] for x in results["trajectory_arr_lafw"]],
    [x[5] for x in results["trajectory_arr_bcg"]],
    results["gd_times"],
]
primal_list = [
    [x[2] - results["ref_primal_value"] for x in results["trajectory_arr_lafw"]],
    [x[2] - results["ref_primal_value"] for x in results["trajectory_arr_bcg"]],
    [x - results["ref_primal_value"] for x in results["function_values_gd"]],
]
test_list = [
    [x[6] for x in results["trajectory_arr_lafw"]],
    [x[6] for x in results["trajectory_arr_bcg"]],
    results["function_values_test_gd"],
]
label = [L"\textrm{L-AFW}", L"\textrm{BCG}", L"\textrm{GD}"]
coefficient_error_values = [
    [x[7] for x in results["trajectory_arr_lafw"]],
    [x[7] for x in results["trajectory_arr_bcg"]],
    results["coefficient_error_gd"],
]


plot_results(
    [primal_list, primal_list, test_list, test_list],
    [iteration_list, time_list, iteration_list, time_list],
    label,
    [L"\textrm{Iteration}", L"\textrm{Time}", L"\textrm{Iteration}", L"\textrm{Time}"],
    [L"\textrm{Primal Gap}", L"\textrm{Primal Gap}", L"\textrm{Test loss}", L"\textrm{Test loss}"],
    xscalelog=[:log, :identity, :log, :identity],
    legend_position=[:bottomleft, nothing, nothing, nothing],
    filename="polynomial_result.pdf",
)
