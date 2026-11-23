module Test_optimal_status

using FrankWolfe
using LinearAlgebra
using Test

# The minimizer (0, 1/2) of f over the box [0, 1]^2 is the midpoint of an edge, and f is more
# curved along that edge, i.e., in direction (0, 1), than in direction (1, 0). A step from an
# iterate whose Frank-Wolfe gap is below epsilon can then end at an iterate whose gap is not.
const H = Diagonal([1.0, 10.0])
const x_opt = [0.0, 0.5]
f(x) = dot(x - x_opt, H * (x - x_opt)) / 2
function grad!(storage, x)
    return storage .= H * (x - x_opt)
end
const lmo = FrankWolfe.BoxLMO(zeros(2), ones(2))
const x0 = [1.0, 0.0]

# `step` is a factory, every run starts from a fresh step object
corrective(step) =
    (; kwargs...) -> FrankWolfe.corrective_frank_wolfe(
        f,
        grad!,
        lmo,
        step(),
        FrankWolfe.ActiveSet([(1.0, x0)]);
        kwargs...,
    )
active_set_method(method; method_kwargs...) =
    (; kwargs...) ->
        method(f, grad!, lmo, FrankWolfe.ActiveSet([(1.0, x0)]); method_kwargs..., kwargs...)
# DICG needs a decomposition-invariant LMO, the 0-1 hypercube is the same box
dicg_method(method; method_kwargs...) =
    (; kwargs...) ->
        method(f, grad!, FrankWolfe.ZeroOneHypercubeLMO(), copy(x0); method_kwargs..., kwargs...)

const SOLVERS = [
    "frank_wolfe" =>
        (; kwargs...) -> FrankWolfe.frank_wolfe(f, grad!, lmo, copy(x0); kwargs...),
    "lazified_conditional_gradient" =>
        (; kwargs...) ->
            FrankWolfe.lazified_conditional_gradient(f, grad!, lmo, copy(x0); kwargs...),
    "away_frank_wolfe" => active_set_method(FrankWolfe.away_frank_wolfe),
    "away_frank_wolfe lazy" => active_set_method(FrankWolfe.away_frank_wolfe, lazy=true),
    "away_frank_wolfe without away steps" =>
        active_set_method(FrankWolfe.away_frank_wolfe, away_steps=false),
    "pairwise_frank_wolfe" => active_set_method(FrankWolfe.pairwise_frank_wolfe),
    "pairwise_frank_wolfe lazy" =>
        active_set_method(FrankWolfe.pairwise_frank_wolfe, lazy=true),
    "blended_pairwise_conditional_gradient" =>
        active_set_method(FrankWolfe.blended_pairwise_conditional_gradient),
    "blended_pairwise_conditional_gradient lazy" =>
        active_set_method(FrankWolfe.blended_pairwise_conditional_gradient, lazy=true),
    "decomposition_invariant_conditional_gradient" =>
        dicg_method(FrankWolfe.decomposition_invariant_conditional_gradient),
    "decomposition_invariant_conditional_gradient lazy" =>
        dicg_method(FrankWolfe.decomposition_invariant_conditional_gradient, lazy=true),
    "blended_decomposition_invariant_conditional_gradient" =>
        dicg_method(FrankWolfe.blended_decomposition_invariant_conditional_gradient),
    "blended_decomposition_invariant_conditional_gradient lazy" =>
        dicg_method(FrankWolfe.blended_decomposition_invariant_conditional_gradient, lazy=true),
    "corrective AwayStep" => corrective(() -> FrankWolfe.AwayStep()),
    "corrective AwayStep lazy" => corrective(() -> FrankWolfe.AwayStep(true)),
    "corrective BlendedPairwiseStep" => corrective(() -> FrankWolfe.BlendedPairwiseStep()),
    "corrective BlendedPairwiseStep lazy" =>
        corrective(() -> FrankWolfe.BlendedPairwiseStep(true)),
    "corrective PairwiseStep" => corrective(() -> FrankWolfe.PairwiseStep()),
    "corrective PairwiseStep lazy" => corrective(() -> FrankWolfe.PairwiseStep(true)),
    "corrective HybridPairAwayStep" =>
        corrective(() -> FrankWolfe.HybridPairAwayStep(false, similar(x0))),
    "corrective HybridPairAwayStep lazy" =>
        corrective(() -> FrankWolfe.HybridPairAwayStep(true, similar(x0))),
    "corrective SimplexGradientDescentStep" =>
        corrective(() -> FrankWolfe.SimplexGradientDescentStep()),
    "corrective SimplexGradientDescentStep lazy" =>
        corrective(() -> FrankWolfe.SimplexGradientDescentStep(true)),
    "corrective ProjectedGradientDescentStep" =>
        corrective(() -> FrankWolfe.ProjectedGradientDescentStep(hessian=Matrix(H))),
    "corrective ProjectedGradientDescentStep lazy" =>
        corrective(() -> FrankWolfe.ProjectedGradientDescentStep(hessian=Matrix(H), lazy=true)),
]

const LINE_SEARCHES =
    (FrankWolfe.Secant, FrankWolfe.Adaptive, FrankWolfe.AdaptiveZerothOrder, FrankWolfe.Goldenratio)

@testset "STATUS_OPTIMAL only with a dual gap below epsilon" begin
    @testset "$name" for (name, run) in SOLVERS
        @testset "$(nameof(line_search)), epsilon = $epsilon" for line_search in LINE_SEARCHES,
            epsilon in (1e-2, 1e-3, 1e-4, 1e-5, 1e-6)

            result = run(; epsilon=epsilon, line_search=line_search())
            if result.status == FrankWolfe.STATUS_OPTIMAL
                @test result.dual_gap <= epsilon
            end
        end
    end
end

# The lazy AwayStep leaves FW steps to corrective_frank_wolfe, which skips them if the FW gap
# is below epsilon. The algorithm has to stop then, instead of repeating the same iteration
# until max_iteration.
@testset "Lazy AwayStep stops when its FW step is skipped" begin
    @testset "epsilon = $epsilon" for epsilon in (1e-3, 1e-4)
        result = corrective(() -> FrankWolfe.AwayStep(true))(; epsilon=epsilon)
        @test result.status == FrankWolfe.STATUS_OPTIMAL
        @test result.dual_gap <= epsilon
    end
end

end # module
