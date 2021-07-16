using LinearAlgebra
using FrankWolfe
using ProgressMeter
using Plots

n = Int(1e2)
k = Int(1e4)
eps=1e-8

xpi = rand(n);
total = sum(xpi);
const xp = xpi ./ total;

# better for memory consumption as we do coordinate-wise ops

function cf(x, xp)
    return LinearAlgebra.norm(x .- xp)^2
end

function cgrad!(storage, x, xp)
    return @. storage = 2 * (x - xp)
end

lmo = FrankWolfe.ProbabilitySimplexOracle(1.0);
# lmo = FrankWolfe.UnitSimplexOracle(1.0);
# lmo = FrankWolfe.UnitSimplexOracle(1.0);
# lmo = FrankWolfe.KSparseLMO(40, 1.0);
# lmo = FrankWolfe.LpNormLMO{2}(1.0)

x00 = FrankWolfe.compute_extreme_point(lmo, zeros(n));

FrankWolfe.benchmark_oracles(
    x -> cf(x, xp),
    (str, x) -> cgrad!(str, x, xp),
    () -> randn(n),
    lmo;
    k=100,
)


function build_callback(storage)
    return function callback(data)
        return push!(storage, (Tuple(data)[1:5]...,data.gamma))
    end
end


####### 2/(2+t) rule

x0 = copy(x00)

trajectory_ag = Vector{Tuple{Int64,Float64,Float64,Float64,Float64,Float64}}()
callback = build_callback(trajectory_ag)

@time x, v, primal, dual_gap = FrankWolfe.frank_wolfe(
    x -> cf(x, xp),
    (str, x) -> cgrad!(str, x, xp),
    lmo,
    x0,
    max_iteration=k,
    trajectory=true,
    epsilon=eps,
    line_search=FrankWolfe.Agnostic(),
    print_iter=k / 10,
    callback=callback,
    emphasis=FrankWolfe.memory,
    verbose=true,
);


####### adaptive

x0 = copy(x00)

trajectory_ada = Vector{Tuple{Int64,Float64,Float64,Float64,Float64,Float64}}()
callback = build_callback(trajectory_ada)

@time x, v, primal, dual_gap = FrankWolfe.frank_wolfe(
    x -> cf(x, xp),
    (str, x) -> cgrad!(str, x, xp),
    lmo,
    x0,
    max_iteration=k,
    trajectory=true,
    epsilon=eps,
    line_search=FrankWolfe.Adaptive(),
    print_iter=k / 10,
    callback=callback,
    emphasis=FrankWolfe.memory,
    verbose=true,
);


####### backtracking

x0 = copy(x00)

trajectory_ls = Vector{Tuple{Int64,Float64,Float64,Float64,Float64,Float64}}()
callback = build_callback(trajectory_ls)

@time x, v, primal, dual_gap = FrankWolfe.frank_wolfe(
    x -> cf(x, xp),
    (str, x) -> cgrad!(str, x, xp),
    lmo,
    x0,
    max_iteration=k,
    trajectory=true,
    epsilon=eps,
    line_search=FrankWolfe.Shortstep(),
    print_iter=k / 10,
    L=2,
    callback=callback,
    emphasis=FrankWolfe.memory,
    verbose=true,
);

x_ag = [trajectory_ag[i][1]+1 for i in eachindex(trajectory_ag)]
gamma_ag = [trajectory_ag[i][6] for i in eachindex(trajectory_ag)]
x_ada = [trajectory_ada[i][1]+1 for i in eachindex(trajectory_ada)]
gamma_ada = [trajectory_ada[i][6] for i in eachindex(trajectory_ada)]
x_ls = [trajectory_ls[i][1]+1 for i in eachindex(trajectory_ls)]
gamma_ls = [trajectory_ls[i][6] for i in eachindex(trajectory_ls)]

Plots.plot(x_ag,gamma_ag,label="gamma_ag", yaxis=:log, xaxis=:log)
Plots.plot!(x_ada,gamma_ada,label="gamma_ada", yaxis=:log, xaxis=:log)
Plots.plot!(x_ls,gamma_ls,label="gamma_ls", yaxis=:log, xaxis=:log)
