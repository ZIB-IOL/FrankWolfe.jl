using LinearAlgebra
using FrankWolfe
using CSV
using DataFrames

include("../examples/plot_utils.jl")

# build feature and outcome vectors
function preprocess_dataframe(df)
    index = df.target
    target_correct_scale = sort!(unique(df.target)) == [-1,1]
    if !target_correct_scale
        @assert(sort!(unique(df.target)) == [1,2])
    end
    nf = size(df, 2) - 3
    a_s = Vector{NTuple{nf, Float64}}()
    ys = Vector{Float64}()
    sizehint!(a_s, size(df, 1))
    sizehint!(ys, size(df, 1))
    for r in eachrow(df)
        if target_correct_scale
            push!(ys, r.target)
        else
            push!(ys, r.target * 2 - 3)
        end
        push!(a_s, values(r[4:end]))
    end
    return (a_s, ys)
end

function build_objective_gradient(df, mu)
    (a_s, ys) = preprocess_dataframe(df)
    # just flexing with unicode
    # reusing notation from Bach 2010 Self-concordant analysis for LogReg
    ℓ(u) = log(exp(u/2) + exp(-u/2))
    dℓ(u) = -1/2 + inv(1 + exp(-u))
    n = length(ys)
    invn = inv(n)
    function f(x)
        err_term = invn * sum(eachindex(ys)) do i
            dtemp = dot(a_s[i], x)
            ℓ(dtemp) - ys[i] * dtemp / 2
        end
        pen_term = mu * dot(x, x) / 2
        err_term + pen_term
    end
    function grad!(storage, x)
        storage .= 0
        for i in eachindex(ys)
            dtemp = dot(a_s[i], x)
            @. storage += invn * a_s[i] * (dℓ(dtemp) - ys[i] / 2)
        end
        @. storage += mu * x
        storage
    end
    (f, grad!)
end

function run_frank_wolfe(df)
    (f0, grad0!) = build_objective_gradient(df,  1/sqrt(size(df, 1)))

    # similar to Frank-Wolfe Newton parameters
    lmo = FrankWolfe.LpNormLMO{1}(1)
    x0 = FrankWolfe.compute_extreme_point(lmo, -ones(length(f0.a_s[1])))
    storage = collect(x0)

    # warning: extremely slow
    (x, v, primal_back, dual_gap, traj_data_backtracking) = FrankWolfe.frank_wolfe(
        f0, grad0!, lmo, x0,
        verbose=true,
        trajectory=true,
        line_search=FrankWolfe.Adaptive(),
        max_iteration=10000,
        gradient=storage,
    )

    (xback, v, primal_back, dual_gap, traj_data_monotonous) = FrankWolfe.frank_wolfe(
        f0, grad0!, lmo, x0,
        verbose=true,
        trajectory=true,
        line_search=MonotonousStepSize(),
        linesearch_tol=1e-8,
        max_iteration=10000,
        gradient=storage,
    )

    (xsecant, v, primal_secant, dual_gap, traj_data_secant) = FrankWolfe.frank_wolfe(
        f0, grad0!, lmo, x0,
        verbose=true,
        trajectory=true,
        line_search=FrankWolfe.Secant(),
        max_iteration=10000,
        gradient=storage,
    )

    return (traj_data_backtracking, traj_data_monotonous, traj_data_secant)
end

# Specify the filename of the instance to load
filename = joinpath(@__DIR__, "data/a1a.csv")  # Change this to the desired file

df = CSV.read(filename, DataFrame)
(traj_data_backtracking, traj_data_monotonous, traj_data_secant) = run_frank_wolfe(df)

# Plot results
data = [traj_data_backtracking, traj_data_monotonous, traj_data_secant]
labels = ["Backtracking", "Monotonous", "Secant"]
plot_trajectories(data, labels, xscalelog=true)
