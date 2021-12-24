using FrankWolfe
using Test
using LinearAlgebra
using DoubleFloats
using DelimitedFiles
import FrankWolfe: ActiveSet

using Profile
using PProf
using PkgBenchmark
# using BenchmarkTools


filename = joinpath(@__DIR__, "lentraj.txt")
len_traj_array = zeros(0)

# Define a parent BenchmarkGroup to contain our SUITE
SUITE = PkgBenchmark.BenchmarkGroup()

# Add some child groups to our benchmark SUITE. The most relevant BenchmarkGroup constructor
# for this case is BenchmarkGroup(tags::Vector). These tags are useful for
# filtering benchmarks by topic, which we'll cover in a later section.
SUITE["vanilla_fw"] = PkgBenchmark.BenchmarkGroup(["step_size", "momentum"])
# SUITE["lazified_cd"] = BenchmarkGroup(["step_size", "cache"])
# SUITE["blas_vs_memory"] = BenchmarkGroup(["blas", "memory"])
# SUITE["rational"] = BenchmarkGroup(["math", "triangles"])
# SUITE["multi_precision"] = BenchmarkGroup(["math", "triangles"])
# SUITE["stochastic_fw"] = BenchmarkGroup(["math", "triangles"])
# SUITE["away_step_fw"] = BenchmarkGroup(["math", "triangles"])
# SUITE["blended_cg"] = BenchmarkGroup(["math", "triangles"])

# macro counted(f)
#     name = f.args[1].args[1]
#     name_str = String(name)
#     body = f.args[2]
#     counter_code = quote
#         if !haskey(COUNTERS, $name_str)
#             COUNTERS[$name_str] = 0
#         end
#         COUNTERS[$name_str] += 1
#     end
#     insert!(body.args, 1, counter_code)
#     return f
# end

mutable struct Counters
    lmo_calls::Int
    grad_calls::Int
    f_calls::Int
end

counters = Counters(0,0,0)

f(x) = norm(x)^2
function grad!(storage, x)
    return storage .= 2x
end
lmo_prob = FrankWolfe.ProbabilitySimplexOracle(1)
x0 = FrankWolfe.compute_extreme_point(lmo_prob, zeros(5))




# x, v, primal, dual_gap, trajectory = FrankWolfe.frank_wolfe(
#                                             f,
#                                             grad!,
#                                             lmo_prob,
#                                             x0,
#                                             max_iteration=1000,
#                                             line_search=FrankWolfe.Agnostic(),
#                                             trajectory=true,
#                                             verbose=false,
#                                         )

# SUITE["vanilla_fw"]["time"] = PkgBenchmark.@benchmarkable FrankWolfe.frank_wolfe(
#                                             f,
#                                             grad!,
#                                             lmo_prob,
#                                             x0,
#                                             max_iteration=1000,
#                                             line_search=FrankWolfe.Agnostic(),
#                                             trajectory=true,
#                                             verbose=false,
#                                             counters = counters
#                                         )
# SUITE["vanilla_fw"]["num_iter"] = trajectory[end][1]
# SUITE["vanilla_fw"]["dual_gap"] = trajectory[end][4]
# SUITE["vanilla_fw"]["lmo_calls"] = trajectory[end][6]

# SUITE["vanilla_fw"]["lmo_calls"] = trajectory[end][6]

# judgement = PkgBenchmark.judge(results,results)

x, v, primal, dual_gap, trajectory = FrankWolfe.frank_wolfe(
                                            f,
                                            grad!,
                                            lmo_prob,
                                            x0,
                                            max_iteration=1000,
                                            line_search=FrankWolfe.Agnostic(),
                                            trajectory=true,
                                            verbose=false,
                                        )


SUITE["vanilla_fw"]["time"] = PkgBenchmark.@benchmarkable FrankWolfe.frank_wolfe(
                                            f,
                                            grad!,
                                            lmo_prob,
                                            x0,
                                            max_iteration=1000,
                                            line_search=FrankWolfe.Agnostic(),
                                            trajectory=true,
                                            verbose=false,
                                        )


                                        


# results = run(SUITE; verbose = true)
# pprof()