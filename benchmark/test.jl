using FrankWolfe
using Test
using LinearAlgebra
using DoubleFloats
using DelimitedFiles
import FrankWolfe: ActiveSet

SUITE = Dict()

SUITE["vanilla_fw"] = Dict()
SUITE["lazified_cd"] = Dict()
SUITE["blas_vs_memory"] = Dict()
SUITE["dense_structure"] = Dict()
SUITE["rational"] = Dict()
SUITE["multi_precision"] = Dict()
SUITE["stochastic_fw"] = Dict()
SUITE["away_step_fw"] = Dict()
SUITE["blended_cg"] = Dict()

# "Testing vanilla Frank-Wolfe with various step size and momentum strategies" begin
    f(x) = norm(x)^2
    function grad!(storage, x)
        return storage .= 2x
    end
    lmo_prob = FrankWolfe.ProbabilitySimplexOracle(1)
    x0 = FrankWolfe.compute_extreme_point(lmo_prob, zeros(5))
SUITE["vanilla_fw"]["1"] = FrankWolfe.frank_wolfe(
            f,
            grad!,
            lmo_prob,
            x0,
            max_iteration=1000,
            line_search=FrankWolfe.Agnostic(),
            trajectory=true,
            verbose=false,
    )
