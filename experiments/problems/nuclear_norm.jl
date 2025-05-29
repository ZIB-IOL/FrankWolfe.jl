using FrankWolfe
import Random
using SparseArrays, LinearAlgebra
using Test

function build_nuclear_norm_problem(seed, dim)
    Random.seed!(seed)
    # rank of the real data
    nobs = dim
    nfeat = dim
    r = max(Int(floor(dim/100)), 1)
    Xreal = Matrix{Float64}(undef, nobs, nfeat)

    X_gen_cols = randn(nfeat, r)
    X_gen_rows = randn(r, nobs)
    svals = 100 * rand(r)
    for i in 1:nobs
        for j in 1:nfeat
            Xreal[i, j] = sum(X_gen_cols[j, k] * X_gen_rows[k, i] * svals[k] for k in 1:r)
        end
    end
    @test rank(Xreal) == r
    # 0.2 of entries missing
    missing_entries = unique!([(rand(1:nobs), rand(1:nfeat)) for _ in 1:dim*10])
    present_entries = [(i, j) for i in 1:nobs, j in 1:nfeat if (i, j) ∉ missing_entries]

    f(X) = 0.5 * sum((X[i, j] - Xreal[i, j])^2 for (i, j) in present_entries, init=0.0)

    function grad!(storage, X)
        storage .= 0
        for (i, j) in present_entries
            storage[i, j] = X[i, j] - Xreal[i, j]
        end
        return nothing
    end

    lmo = FrankWolfe.NuclearNormLMO(275_000.0)
    x0 = FrankWolfe.compute_extreme_point(lmo, zero(Xreal))
    active_set = FrankWolfe.ActiveSet([(1.0, x0)])

    return f, grad!, lmo, x0, active_set, x -> true, dim^2
end

