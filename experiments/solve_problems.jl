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

@enum LineSearchVariant begin
    LS_ONLY_SECANT = 1
    LS_SECANT_WITH_BACKTRACKING = 2
    LS_BACKTRACKING_AND_SECANT = 3
    LS_ADAPTIVE = 4
end

const linesearchvariant_string = (
    LS_ONLY_SECANT ="Only_Secant",
    LS_SECANT_WITH_BACKTRACKING="Secant_with_Backtracking",
    LS_BACKTRACKING_AND_SECANT="Backtracking_and_Secant",
    LS_ADAPTIVE="Adaptive",
)

function solve_problems(seed, dimension, problem, ls_variant; time_limit=3600, write=true, verbose=true, FW_variant="BPCG", max_iter=Inf)
    f, grad!, lmo, x0, active_set, domain_oracle = if problem == "OEDP_A"
        build_optimal_design(seed, dimension^2, criterion="A")
    elseif problem == "OEDP_D"
        build_optimal_design(seed, dimension^2, criterion="D")
    elseif problem == "Nuclear"
        build_nuclear_norm_problem(seed, dimension)
    elseif problem == "Birkhoff"
        build_birkhoff_problem(seed, dimension)
    elseif problem == "QuadraticProbSimplex"
        build_simple_self_concordant_problem(seed, dimension^2)
    elseif problem == "Spectrahedron"
        build_spectrahedron(seed, dimension)
    elseif problem == "IllConditionedQuadratic"
        build_ill_conditioned_quadratic(seed, dimension^2)
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
        FrankWolfe.BacktrackingAndSecant(domain_oracle=domain_oracle)
    elseif ls_variant == LS_SECANT_WITH_BACKTRACKING
        FrankWolfe.Secant(safe=true, domain_oracle=domain_oracle)
    elseif ls_variant == LS_ADAPTIVE
        FrankWolfe.Adaptive(domain_oracle=domain_oracle)
    end
    # Precompile run
    fw_variant(f, grad!, lmo, active_set, line_search=line_search, timeout=10, max_iteration=max_iter)

    # Actual run
    data = @timed fw_variant(f, grad!, lmo, active_set, line_search=line_search, timeout=time_limit, max_iteration=max_iter, verbose=verbose, trajectory=true)

    @show data.value.primal, data.value.dual_gap
    @show data.value.traj_data[end][1], data.value.traj_data[end][end]
    #@show data.value.x 
    #@show data.value.v
    #@show data.value.active_set

    if write
        df = DataFrame(seed=seed, dimension=dimension^2, time=data.time, fw_time=data.value.traj_data[end][end], primal = data.value.primal, dual_gap=data.value.dual_gap, iterations=data.value.traj_data[end][1])
        
        file_name = joinpath(@__DIR__, "csv/" * problem * "/" * string(ls_variant) * string(dimension^2) * "_" * string(seed) * ".csv")
        CSV.write(file_name, df, append=false, writeheader=true)
    end

    return data.value.x, data.value.primal, data.value.dual_gap
end
