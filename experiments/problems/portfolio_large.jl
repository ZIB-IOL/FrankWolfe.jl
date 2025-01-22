using LinearAlgebra
using JSON
using DelimitedFiles
using MAT

function build_portfolio(seed, dimension)
    # Read the matrx file
    problem_instance = if seed in [1,2,3] && dimension in [800, 1200, 1500]
        joinpath(@__DIR__, "portfolio_data/syn_1000_$(dimension)_10_50_$(seed).mat")
    elseif seed == 4 && dimension in [800, 1200, 1500]
        joinpath(@__DIR__, "portfolio_data/syn_1000_$(dimension)_10_50.mat")
    elseif seed == 5 && dimension == 1500
        joinpath(@__DIR__, "portfolio_data/syn_1500_1500.mat")
    elseif seed == 1 && dimension in [2000, 5000]
        joinpath(@__DIR__, "portfolio_data/syn_5000_$(dimension).mat")
    elseif seed == 2 && dimension == 2000
        joinpath(@__DIR__, "portfolio_data/synlog_1000_1500.mat")
    elseif seed == 2 && dimension == 5000
        joinpath(@__DIR__, "portfolio_data/synlog_5000_2000.mat")
    else
        error("Problem: $(problem) Seed: $(seed) Dimension: $(dimension)")
    end   

    W = MAT.matread(problem_instance)["W"]

    n, p = size(W)

    # lower bound on objective value
    true_obj_value = -10.0

    function f(x)
        return -sum(log(dot(x, @view(W[:, t]))) for t in 1:p) - true_obj_value
    end

    function grad!(storage, x)
        storage .= 0
        for t in 1:p
            temp_rev = dot(x, @view(W[:, t]))
            @. storage -= @view(W[:, t]) ./ temp_rev
        end
        return storage
    end

    lmo = FrankWolfe.ProbabilitySimplexOracle(1.0)
    x0 = FrankWolfe.compute_extreme_point(lmo, zeros(n))
    active_set = FrankWolfe.ActiveSet([(1.0, x0)])

    return f, grad!, lmo, x0, active_set, x -> true, p
end
#=
problem_instance = joinpath(@__DIR__, "portfolio_data/syn_1000_800_10_50_1.mat")
W1 = MAT.matread(problem_instance)["W"]
n, p = size(W1)
@show n, p

problem_instance = joinpath(@__DIR__, "portfolio_data/syn_1000_800_10_50_2.mat")
W2 = MAT.matread(problem_instance)["W"]
n, p = size(W2)
@show n, p

problem_instance = joinpath(@__DIR__, "portfolio_data/syn_1000_800_10_50_3.mat")
W3 = MAT.matread(problem_instance)["W"]
n, p = size(W3)
@show n, p

problem_instance = joinpath(@__DIR__, "portfolio_data/syn_1000_800_10_50.mat")
W = MAT.matread(problem_instance)["W"]
n, p = size(W)
@show n, p

@show W==W1, W==W2, W==W3

problem_instance = joinpath(@__DIR__, "portfolio_data/syn_1000_1200_10_50_1.mat")
W = MAT.matread(problem_instance)["W"]
n, p = size(W)
@show n, p

problem_instance = joinpath(@__DIR__, "portfolio_data/syn_1000_1200_10_50_2.mat")
W = MAT.matread(problem_instance)["W"]
n, p = size(W)
@show n, p

problem_instance = joinpath(@__DIR__, "portfolio_data/syn_1000_1200_10_50_3.mat")
W = MAT.matread(problem_instance)["W"]
n, p = size(W)
@show n, p

problem_instance = joinpath(@__DIR__, "portfolio_data/syn_1000_1200_10_50.mat")
W = MAT.matread(problem_instance)["W"]
n, p = size(W)
@show n, p

problem_instance = joinpath(@__DIR__, "portfolio_data/syn_1000_1500_10_50_1.mat")
W = MAT.matread(problem_instance)["W"]
n, p = size(W)
@show n, p

problem_instance = joinpath(@__DIR__, "portfolio_data/syn_1000_1500_10_50_2.mat")
W = MAT.matread(problem_instance)["W"]
n, p = size(W)
@show n, p

problem_instance = joinpath(@__DIR__, "portfolio_data/syn_1000_1500_10_50_3.mat")
W = MAT.matread(problem_instance)["W"]
n, p = size(W)
@show n, p

problem_instance = joinpath(@__DIR__, "portfolio_data/syn_1000_1500_10_50.mat")
W = MAT.matread(problem_instance)["W"]
n, p = size(W)
@show n, p

problem_instance = joinpath(@__DIR__, "portfolio_data/syn_1500_1500.mat")
W = MAT.matread(problem_instance)["W"]
n, p = size(W)
@show n, p

problem_instance = joinpath(@__DIR__, "portfolio_data/syn_5000_2000.mat")
W = MAT.matread(problem_instance)["W"]
n, p = size(W)
@show n, p

problem_instance = joinpath(@__DIR__, "portfolio_data/syn_5000_5000.mat")
W = MAT.matread(problem_instance)["W"]
n, p = size(W)
@show n, p

problem_instance = joinpath(@__DIR__, "portfolio_data/synlog_1000_1500.mat")
W = MAT.matread(problem_instance)["W"]
n, p = size(W)
@show n, p

problem_instance = joinpath(@__DIR__, "portfolio_data/synlog_5000_2000.mat")
W = MAT.matread(problem_instance)["W"]
n, p = size(W)
@show n, p =#




