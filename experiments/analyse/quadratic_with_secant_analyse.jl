using FrankWolfe
using Test
using DataFrames
using CSV


for file in readdir(joinpath(@__DIR__, "problems/"), join=true)
    if endswith(file, "jl")
        include(file)
    end
end

analyse_data_normal = []
analyse_data_ill = []
limit_num_steps_secant=40
file_name_normal = joinpath(@__DIR__, "normal_quadratic_with_secant_analsye_data.csv")
file_name_ill = joinpath(@__DIR__, "ill_quadratic_with_secant_analsye_data.csv")
for m in [100, 200, 300, 400, 500]
    for seed in 1:10
        Random.seed!(seed)
        @show seed, m
    @testset "Normal" begin 
        f, grad!, lmo, x0, active_set, _ = build_simple_self_concordant_problem(seed, m)
        line_search = FrankWolfe.Secant()
        x, _, primal, dual_gap, traj_data = FrankWolfe.blended_pairwise_conditional_gradient(f, grad!, lmo, x0, verbose=true, line_search=line_search, trajectory=true)

        #@show line_search.number_not_converging, traj_data_s[end][1], line_search.number_not_converging/traj_data_s[end][1]
        #@show line_search.best_improvement_by_backtracking, line_search.max_violation

        df = DataFrame(seed=seed, dimension=m, fw_iterations=traj_data[end][1], all_secant_iter=line_search.inner_iter, average_secant_iter=line_search.inner_iter/traj_data[end][1])
        CSV.write(file_name_normal, df, append=true, writeheader= m == 100 && seed==1)

        #push!(analyse_data_A, (seed, m, line_search.number_not_converging, traj_data_s[end][1], line_search.number_not_converging/traj_data_s[end][1], line_search.best_improvement_by_backtracking)) 
    end

    @testset "Ill Conditioned" begin
        f, grad!, lmo, x0, active_set, _ = build_simple_self_concordant_problem(seed, m)
        line_search = FrankWolfe.Secant()
        x, _, primal, dual_gap, traj_data = FrankWolfe.blended_pairwise_conditional_gradient(f, grad!, lmo, x0, verbose=true, line_search=line_search, trajectory=true)

        #@show line_search.number_not_converging, traj_data_s[end][1], line_search.number_not_converging/traj_data_s[end][1]
        #@show line_search.best_improvement_by_backtracking, line_search.max_violation

        df = DataFrame(seed=seed, dimension=m, fw_iterations=traj_data[end][1], all_secant_iter=line_search.inner_iter, average_secant_iter=line_search.inner_iter/traj_data[end][1])
        CSV.write(file_name_ill, df, append=true, writeheader= m == 100 && seed==1)

    end
end
end

df_normal = DataFrame(CSV.File(file_name_normal))

print(df_normal)

println("\n")

df_ill = DataFrame(CSV.File(file_name_ill))

print(df_ill)

println("\n")