using FrankWolfe
using ProgressMeter
using Arpack
using Plots
using DoubleFloats
using ReverseDiff

import Random
using SparseArrays, LinearAlgebra
using Test
using Plots

const nfeat = 100 * 5
const nobs = 500

# rank of the real data
const r = 30
const Xreal = Matrix{Float64}(undef, nobs, nfeat)

const X_gen_cols = randn(nfeat, r)
const X_gen_rows = randn(r, nobs)
const svals = 100 * rand(r)
for i in 1:nobs
    for j in 1:nfeat
        Xreal[i, j] = sum(X_gen_cols[j, k] * X_gen_rows[k, i] * svals[k] for k in 1:r)
    end
end

nucnorm(Xmat) = sum(abs(σi) for σi in LinearAlgebra.svdvals(Xmat))


@test rank(Xreal) == r

# 0.2 of entries missing
const missing_entries = unique!([(rand(1:nobs), rand(1:nfeat)) for _ in 1:10000])
const present_entries = [(i, j) for i in 1:nobs, j in 1:nfeat if (i, j) ∉ missing_entries]

f(X) = 0.5 * sum((X[i, j] - Xreal[i, j])^2 for (i, j) in present_entries)

function grad!(storage, X)
    storage .= 0
    for (i, j) in present_entries
        storage[i, j] = X[i, j] - Xreal[i, j]
    end
    return nothing
end


const lmo = FrankWolfe.NuclearNormLMO(275_000.0)
const x0 = FrankWolfe.compute_extreme_point(lmo, zero(Xreal))

FrankWolfe.benchmark_oracles(f, grad!, () -> randn(size(Xreal)), lmo; k=100)

# gradient descent
gradient = similar(x0)
xgd = Matrix(x0)
for _ in 1:5000
    @info f(xgd)
    grad!(gradient, xgd)
    xgd .-= 0.01 * gradient
    if norm(gradient) ≤ sqrt(eps())
        break
    end
end

grad!(gradient, x0)
v0 = FrankWolfe.compute_extreme_point(lmo, gradient)
@test dot(v0 - x0, gradient) < 0

const k = 500

x00 = copy(x0)

xfin, vmin, _, _, traj_data = FrankWolfe.frank_wolfe(
    f,
    grad!,
    lmo,
    x00;
    epsilon=1e7,
    max_iteration=k,
    print_iter=k / 10,
    trajectory=true,
    verbose=true,
    line_search=FrankWolfe.Adaptive(),
    memory_mode=FrankWolfe.InplaceEmphasis(),
    gradient=spzeros(size(x0)...),
)

xfinlcg, vmin, _, _, traj_data = FrankWolfe.lazified_conditional_gradient(
    f,
    grad!,
    lmo,
    x00;
    epsilon=1e7,
    max_iteration=k,
    print_iter=k / 10,
    trajectory=true,
    verbose=true,
    line_search=FrankWolfe.Adaptive(),
    memory_mode=FrankWolfe.InplaceEmphasis(),
    gradient=spzeros(size(x0)...),
)


x00 = copy(x0)

xfinAFW, vmin, _, _, traj_data = FrankWolfe.away_frank_wolfe(
    f,
    grad!,
    lmo,
    x00;
    epsilon=1e7,
    max_iteration=k,
    print_iter=k / 10,
    trajectory=true,
    verbose=true,
    lazy=true,
    line_search=FrankWolfe.Adaptive(),
    memory_mode=FrankWolfe.InplaceEmphasis(),#,
)

x00 = copy(x0)

xfinBCG, vmin, _, _, traj_data = FrankWolfe.blended_conditional_gradient(
    f,
    grad!,
    lmo,
    x00;
    epsilon=1e7,
    max_iteration=k,
    print_iter=k / 10,
    trajectory=true,
    verbose=true,
    line_search=FrankWolfe.Adaptive(),
    memory_mode=FrankWolfe.InplaceEmphasis(),
)

xfinBPCG, vmin, _, _, traj_data = FrankWolfe.blended_pairwise_conditional_gradient(
    f,
    grad!,
    lmo,
    x00;
    epsilon=1e7,
    max_iteration=k,
    print_iter=k / 10,
    trajectory=true,
    verbose=true,
    line_search=FrankWolfe.Adaptive(),
    memory_mode=FrankWolfe.InplaceEmphasis(),
#    lazy=true,
)

pit = plot(svdvals(xfin), label="FW", width=3, yaxis=:log)
plot!(svdvals(xfinlcg), label="LCG", width=3, yaxis=:log)
plot!(svdvals(xfinAFW), label="LAFW", width=3, yaxis=:log)
plot!(svdvals(xfinBCG), label="BCG", width=3, yaxis=:log)
plot!(svdvals(xfinBPCG), label="BPCG", width=3, yaxis=:log)
plot!(svdvals(xgd), label="Gradient descent", width=3, yaxis=:log)
plot!(svdvals(Xreal), label="Real matrix", linestyle=:dash, width=3, color=:black)
title!("Singular values")

savefig(pit, "matrix_completion.pdf")
