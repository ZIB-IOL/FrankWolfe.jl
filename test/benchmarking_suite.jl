using FrankWolfe
using Test
using LinearAlgebra
using DelimitedFiles
using SparseArrays
using LibGit2
import FrankWolfe: ActiveSet

function run_benchmark()
    f(x) = norm(x)^2

    function grad!(storage, x)
        @. storage = 2x
    end

    lmo = FrankWolfe.ProbabilitySimplexOracle(1)

    tf = FrankWolfe.TrackingObjective(f,0)
    tgrad! = FrankWolfe.TrackingGradient(grad!,0)
    tlmo = FrankWolfe.TrackingLMO(lmo)

    x0 = FrankWolfe.compute_extreme_point(tlmo, spzeros(1000))
    callback = FrankWolfe.TrackingCallback()

    FrankWolfe.frank_wolfe(
        tf,
        tgrad!,
        tlmo,
        x0,
        line_search=FrankWolfe.Agnostic(),
        max_iteration=5000,
        trajectory=true,
        callback=callback,
        verbose=true,
    )
    return callback.storage
end