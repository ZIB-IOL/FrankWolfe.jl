using FrankWolfe
using Test
using LinearAlgebra

n = 10
f(x) = norm(x)^2
function grad!(storage, x)
    @. storage = 2x
    return nothing
end
lmo_prob = FrankWolfe.ProbabilitySimplexOracle(4)
lmo = FrankWolfe.TrackingLMO(lmo_prob)
x0 = FrankWolfe.compute_extreme_point(lmo_prob, randn(10))

FrankWolfe.blended_pairwise_conditional_gradient(
    f,
    grad!,
    lmo,
    x0,
    max_iteration=6000,
    line_search=FrankWolfe.Adaptive(),
    verbose=true,
)
