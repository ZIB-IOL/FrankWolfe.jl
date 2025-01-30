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

function build_linesearch(ls_variant, domain_oracle)
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
    elseif ls_variant == LS_MONOTONIC
        FrankWolfe.MonotonicStepSize(domain_oracle)
    elseif ls_variant == LS_AGNOSTIC    
        FrankWolfe.Agnostic()
    elseif ls_variant == LS_ADAPTIVE_ZERO
        FrankWolfe.AdaptiveZerothOrder(domain_oracle=domain_oracle)
    elseif ls_variant == LS_BACKTRACKING
        FrankWolfe.Backtracking(domain_oracle=domain_oracle,tol=1e-8)
    elseif ls_variant == LS_GOLDEN_RATIO
        FrankWolfe.Goldenratio(domain_oracle)
    else
        error("Line search variant not known.")
    end
    return line_search
end

function build_function_data(problem, seed, dimension)
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

    return f, grad!, lmo, x0, active_set, domain_oracle, dim
end

function solve_problems(seed, dimension, problem, ls_variant; time_limit=3600, write=true, verbose=true, FW_variant="BPCG", max_iter=Inf, print_iter=1000)
    f, grad!, lmo, x0, active_set, domain_oracle, dim = build_function_data(problem, seed, dimension)

    fw_variant = if FW_variant == "BPCG"
        FrankWolfe.blended_pairwise_conditional_gradient
    elseif FW_variant == "Vanilla"
        FrankWolfe.away_frank_wolfe
    else
        error("Frank-Wolfe variant not known.")
    end

    # Set the line search for precompiling
    line_search = build_linesearch(ls_variant, domain_oracle)
    store_step_sizes= is_type_secant(ls_variant) || ls_variant == LS_ADAPTIVE
    # Precompile run
    println("PRECOMPILATION")
    if FW_variant == "BPCG"
        @show "BPCG"
        fw_variant(f, grad!, lmo, active_set, line_search=line_search, timeout=30, max_iteration=max_iter, verbose=verbose, trajectory=true, print_iter=print_iter, store_step_sizes=store_step_sizes, lazy=false)
    else
        @show "Vanilla"
        fw_variant(f, grad!, lmo, active_set, line_search=line_search, timeout=30, max_iteration=max_iter, store_step_sizes=store_step_sizes, lazy=false, away_steps=false)
    end

    # Set line search again to avoid carry over issues from the first run
    line_search = build_linesearch(ls_variant, domain_oracle)
    f, grad!, lmo, x0, active_set, domain_oracle, dim = build_function_data(problem, seed, dimension)
    # Actual run
    println("\nACTUAL RUN")
    data = if FW_variant == "BPCG" 
        @timed fw_variant(f, grad!, lmo, active_set, line_search=line_search, timeout=time_limit, max_iteration=max_iter, verbose=verbose, trajectory=true, print_iter=print_iter, store_step_sizes=store_step_sizes, lazy=false)
    else
        @timed fw_variant(f, grad!, lmo, active_set, line_search=line_search, timeout=time_limit, max_iteration=max_iter, verbose=verbose, trajectory=true, print_iter=print_iter, store_step_sizes=store_step_sizes, lazy=false,away_steps=false)
    end
    smallest_dual_gap = if data.value.traj_data[end][1] != 0
        data.value.traj_data[end-2][end-1]
    else
        0.0
    end
@show data.time
    @show data.value.primal, data.value.dual_gap
    @show data.value.traj_data[end][1], data.value.traj_data[end][end]
    @show smallest_dual_gap
    #@show data.value.x 
    #@show data.value.v
    #@show data.value.active_set

    if write
        # trajectory
        sub_dir = FW_variant == "BPCG" ? "" : FW_variant
        df_traj = DataFrame(data.value.traj_data)
        rename!(df_traj, Dict(1 => "iterations", 2 => "primal", 3 => "dual_bound", 4 => "dual_gap", 5 => "time"))
        if is_type_secant(ls_variant) || ls_variant == LS_ADAPTIVE
            last_gamma = isempty(line_search.step_sizes) ? 0.0 : line_search.step_sizes[end]
            @show length(line_search.step_sizes), length(df_traj[!, :iterations])
            df_traj[!, :step_sizes] = vcat(line_search.step_sizes, last_gamma, last_gamma)
        end
        file_name_traj = joinpath(@__DIR__, "csv/" * sub_dir * "/" * problem * "/trajectory/" * string(ls_variant) * "_" * string(dim) * "_" * string(seed) * ".csv")
        CSV.write(file_name_traj, df_traj, append=false, writeheader=true)

        df = DataFrame()
        if is_type_secant(ls_variant)
            mean_iter = mean(line_search.inner_iter)
            std_iter = std(line_search.inner_iter)

            mean_gap = problem == "Nuclear" && dimension >= 800 ? geom_shifted_mean(line_search.gaps, shift=1e-5) : geom_shifted_mean(line_search.gaps, shift=1e-8)
            std_gap = geo_standard_deviation(line_search.gaps, mean_gap)

            df = DataFrame(seed=seed, 
                dimension=dim, 
                time=data.time, 
                fw_time=data.value.traj_data[end][end], 
                primal = data.value.primal, 
                dual_gap=data.value.dual_gap, 
                smallest_dual_gap=smallest_dual_gap, 
                iterations=data.value.traj_data[end][1],
                iter_not_converging = line_search.number_not_converging,
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
        
        file_name = joinpath(@__DIR__, "csv/" * sub_dir * "/" * problem * "/" * string(ls_variant) * "_" * string(dim) * "_" * string(seed) * ".csv")
        CSV.write(file_name, df, append=false, writeheader=true)
    end

    return data.value.x, data.value.primal, data.value.dual_gap
end
