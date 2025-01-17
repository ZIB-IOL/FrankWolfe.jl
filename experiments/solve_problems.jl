using FrankWolfe
using LinearAlgebra
using DataFrames
using CSV
using Random

for file in readdir(joinpath(@__DIR__, "problems/"), join=true)
    if endswith(file, "jl")
        include(file)
    end
end

include("utilities.jl")

function solve_problems(seed, dimension, problem, ls_variant; time_limit=3600, write=true, verbose=true, FW_variant="BPCG", max_iter=Inf)
    f, grad!, lmo, x0, active_set, domain_oracle, dim = if problem == "OEDP_A"
        build_optimal_design(seed, dimension, criterion="A")
    elseif problem == "OEDP_D"
        build_optimal_design(seed, dimension, criterion="D")
    elseif problem == "Nuclear"
        build_nuclear_norm_problem(seed, dimension)
    elseif problem == "Birkhoff"
        build_birkhoff_problem(seed, dimension)
    elseif problem == "QuadraticProbSimplex"
        build_simple_self_concordant_problem(seed, dimension^2)
    elseif problem == "Spectrahedron"
        build_spectrahedron(seed, dimension)
    elseif problem == "IllConditionedQuadratic"
        build_ill_conditioned_quadratic(seed, dimension)
    elseif problem == "Portfolio"
        build_portfolio(seed, dimension)
    else
        error("Problem type not known.")
    end

    fw_variant = if FW_variant == "BPCG"
        FrankWolfe.blended_pairwise_conditional_gradient
    else
        error("Frank-Wolfe variant not known.")
    end

    # Set the line search
    line_search = if ls_variant == LS_ONLY_SECANT
        FrankWolfe.Secant(safe=false, domain_oracle=domain_oracle)
    elseif ls_variant == LS_BACKTRACKING_AND_SECANT
        FrankWolfe.ImprovedGammaSecant(first_ls=FrankWolfe.Backtracking(),domain_oracle=domain_oracle)
    elseif ls_variant == LS_SECANT_WITH_BACKTRACKING
        FrankWolfe.Secant(safe=true, domain_oracle=domain_oracle)
    elseif ls_variant == LS_ADAPTIVE
        FrankWolfe.Adaptive(domain_oracle=domain_oracle)
    elseif ls_variant == LS_ADAPTIVE_AND_SECANT
        FrankWolfe.ImprovedGammaSecant(first_ls=FrankWolfe.Adaptive(), domain_oracle=domain_oracle)
    elseif ls_variant == LS_ADAPTIVE_ZERO_AND_SECANT
        FrankWolfe.ImprovedGammaSecant(first_ls=FrankWolfe.AdaptiveZerothOrder(), domain_oracle=domain_oracle)
    elseif ls_variant == LS_SECANT_3
        FrankWolfe.Secant(safe=false, domain_oracle=domain_oracle, limit_num_steps=3)
    elseif ls_variant == LS_SECANT_5
        FrankWolfe.Secant(safe=false, domain_oracle=domain_oracle, limit_num_steps=5)
    elseif ls_variant == LS_SECANT_7
        FrankWolfe.Secant(safe=false, domain_oracle=domain_oracle, limit_num_steps=7)
    elseif ls_variant == LS_SECANT_12
        FrankWolfe.Secant(safe=false, domain_oracle=domain_oracle, limit_num_steps=12)
    else
        error("Line search variant not known.")
    end
    # Precompile run
    fw_variant(f, grad!, lmo, active_set, line_search=line_search, timeout=10, max_iteration=max_iter)

    # Actual run
    data = @timed fw_variant(f, grad!, lmo, active_set, line_search=line_search, timeout=time_limit, max_iteration=max_iter, verbose=verbose, trajectory=true)
    smallest_dual_gap = if data.value.traj_data[end][1] != 0
        data.value.traj_data[end-2][end-1]
    else
        0.0
    end

    @show data.value.primal, data.value.dual_gap
    @show data.value.traj_data[end][1], data.value.traj_data[end][end]
    @show smallest_dual_gap
    #@show data.value.x 
    #@show data.value.v
    #@show data.value.active_set

    if write
        # trajectory
        df_traj = DataFrame(data.value.traj_data)
        rename!(df_traj, Dict(1 => "iterations", 2 => "primal", 3 => "dual_bound", 4 => "dual_gap", 5 => "time"))
        file_name_traj = joinpath(@__DIR__, "csv/" * problem * "/trajectory/" * string(ls_variant) * "_" * string(dim) * "_" * string(seed) * ".csv")
        CSV.write(file_name_traj, df_traj, append=false, writeheader=true)

        df = DataFrame()
        if is_type_secant(ls_variant)
            mean_iter = mean(line_search.inner_iter)
            std_iter = std(line_search.inner_iter)

            mean_gap = geom_shifted_mean(line_search.gap, shift=1e-8)
            std_gap = geo_standard_deviation(line_search.gap, mean_gap)

            df = DataFrame(seed=seed, 
                dimension=dim, 
                time=data.time, 
                fw_time=data.value.traj_data[end][end], 
                primal = data.value.primal, 
                dual_gap=data.value.dual_gap, 
                smallest_dual_gap=smallest_dual_gap, 
                iterations=data.value.traj_data[end][1],
                iter_not_converging = line_search.iter_not_converging,
                fallback_helped = line_search.fallback_help,
                average_iter = mean_iter,
                std_iter = std_iter,
                average_gap = mean_gap,
                std_gap = std_gap
            )
        elseif ls_variant == LS_ADAPTIVE
            mean_iter = mean(line_search.number_itertions)
            std_iter = std(line_search.number_itertions)
            df = DataFrame(seed=seed, 
                dimension=dim, 
                time=data.time, 
                fw_time=data.value.traj_data[end][end], 
                primal = data.value.primal, 
                dual_gap=data.value.dual_gap, 
                smallest_dual_gap=smallest_dual_gap, 
                iterations=data.value.traj_data[end][1],
                average_iter = mean_iter,
                std_iter = std_iter
            )
        else
            df = DataFrame(seed=seed, 
                dimension=dim, 
                time=data.time, 
                fw_time=data.value.traj_data[end][end], 
                primal = data.value.primal, 
                dual_gap=data.value.dual_gap, 
                smallest_dual_gap=smallest_dual_gap, 
                iterations=data.value.traj_data[end][1]
            )
        end
        
        file_name = joinpath(@__DIR__, "csv/" * problem * "/" * string(ls_variant) * "_" * string(dim) * "_" * string(seed) * ".csv")
        CSV.write(file_name, df, append=false, writeheader=true)
    end

    return data.value.x, data.value.primal, data.value.dual_gap
end
