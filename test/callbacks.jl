module Test_callbacks

using FrankWolfe
using LinearAlgebra
using Test
using StableRNGs

using FrankWolfe: ST_REGULAR, ST_LAZY, ST_DUALSTEP, ST_AWAY, ST_PAIRWISE, ST_DROP
using FrankWolfe: ST_SIMPLEXDESCENT, ST_LAST, ST_POSTPROCESS

const n = 20
# the minimizer of f over the probability simplex lies on a low-dimensional face,
# which forces away, drop, and pairwise steps in the active-set based variants
const xp = rand(StableRNG(1234), n) .- 0.4
# non-uniform curvature, so that the accelerated BCG inner solver is used
const D = 1 .+ 2 .* rand(StableRNG(5678), n)
f(x) = sum(D .* (x .- xp) .^ 2)
function grad!(storage, x)
    @. storage = 2 * D * (x - xp)
    return nothing
end
const hessian = Matrix(Diagonal(2 .* D))
const lmo = FrankWolfe.ProbabilitySimplexLMO(1.0)
const x0 = FrankWolfe.compute_extreme_point(lmo, zeros(n))
# kwargs common to all deterministic algorithms
const KW = (max_iteration=2000, epsilon=1e-9)

# finite-sum objective for the stochastic Frank-Wolfe
const data = let rng = StableRNG(42)
    [xp .+ 0.05 .* randn(rng, n) for _ in 1:50]
end
const f_stoch = FrankWolfe.StochasticObjective(
    (θ, x) -> norm(θ - x)^2,
    (storage, θ, x) -> (@. storage += 2 * (θ - x)),
    data,
    zeros(n),
)

struct CallRecord
    step_type::FrankWolfe.StepType
    x::Vector{Float64}
    grad_ok::Bool
end

"""
Callback recording every in-loop call and returning `false` at the `stop_at`-th one.
`objective` is the objective passed to the algorithm. It identifies the states of the
BCG inner solvers over the reduced problem, whose `x` are the active set weights.
"""
function recording_callback(records, objective; stop_at=0)
    return function callback(state, args...)
        if state.step_type in (ST_LAST, ST_POSTPROCESS)
            return true
        end
        x = if state.f === objective
            collect(state.x)
        else
            active_set = args[1]
            y = zeros(n)
            for (λ, atom) in zip(state.x, active_set.atoms)
                @. y += λ * atom
            end
            y
        end
        # the gradient was computed at the reported iterate, i.e., before the step
        grad_ok = if state.grad! === nothing
            true
        else
            g = similar(state.gradient)
            state.grad!(g, state.x)
            isapprox(g, state.gradient, rtol=1e-6, atol=1e-10)
        end
        push!(records, CallRecord(state.step_type, x, grad_ok))
        return length(records) != stop_at
    end
end

# `step` is a factory: every run must start from a fresh step object so that the interrupted
# reruns replay the reference run, and some steps keep state across solver calls.
corrective(step) =
    cb -> FrankWolfe.corrective_frank_wolfe(
        f,
        grad!,
        lmo,
        step(),
        FrankWolfe.ActiveSet([(1.0, x0)]);
        callback=cb,
        KW...,
    )

# (name, objective, run(callback), step types occurring in an uninterrupted run)
const SOLVERS = [
    (
        "frank_wolfe",
        f,
        cb -> FrankWolfe.frank_wolfe(f, grad!, lmo, x0; callback=cb, KW...),
        [ST_REGULAR],
    ),
    (
        "lazified_conditional_gradient",
        f,
        cb -> FrankWolfe.lazified_conditional_gradient(f, grad!, lmo, x0; callback=cb, KW...),
        [ST_LAZY, ST_DUALSTEP],
    ),
    (
        "away_frank_wolfe",
        f,
        cb -> FrankWolfe.away_frank_wolfe(f, grad!, lmo, x0; callback=cb, KW...),
        [ST_REGULAR, ST_AWAY, ST_DROP],
    ),
    (
        "away_frank_wolfe lazy",
        f,
        cb -> FrankWolfe.away_frank_wolfe(f, grad!, lmo, x0; lazy=true, callback=cb, KW...),
        [ST_REGULAR, ST_LAZY, ST_AWAY, ST_DROP, ST_DUALSTEP],
    ),
    (
        "away_frank_wolfe without away steps",
        f,
        cb -> FrankWolfe.away_frank_wolfe(f, grad!, lmo, x0; away_steps=false, callback=cb, KW...),
        [ST_REGULAR],
    ),
    (
        "pairwise_frank_wolfe",
        f,
        cb -> FrankWolfe.pairwise_frank_wolfe(f, grad!, lmo, x0; callback=cb, KW...),
        [ST_PAIRWISE, ST_DUALSTEP],
    ),
    (
        "pairwise_frank_wolfe lazy",
        f,
        cb -> FrankWolfe.pairwise_frank_wolfe(f, grad!, lmo, x0; lazy=true, callback=cb, KW...),
        [ST_PAIRWISE, ST_LAZY, ST_DUALSTEP],
    ),
    (
        "blended_pairwise_conditional_gradient",
        f,
        cb -> FrankWolfe.blended_pairwise_conditional_gradient(
            f,
            grad!,
            lmo,
            x0;
            callback=cb,
            KW...,
        ),
        [ST_REGULAR, ST_PAIRWISE, ST_DROP],
    ),
    (
        "blended_pairwise_conditional_gradient lazy",
        f,
        cb -> FrankWolfe.blended_pairwise_conditional_gradient(
            f,
            grad!,
            lmo,
            x0;
            lazy=true,
            callback=cb,
            KW...,
        ),
        [ST_REGULAR, ST_PAIRWISE, ST_DROP, ST_DUALSTEP],
    ),
    (
        "blended_conditional_gradient",
        f,
        cb -> FrankWolfe.blended_conditional_gradient(f, grad!, lmo, x0; callback=cb, KW...),
        [ST_REGULAR, ST_SIMPLEXDESCENT, ST_DUALSTEP],
    ),
    (
        "blended_conditional_gradient with hessian",
        f,
        cb -> FrankWolfe.blended_conditional_gradient(
            f,
            grad!,
            lmo,
            x0;
            hessian=hessian,
            callback=cb,
            KW...,
        ),
        [ST_REGULAR, ST_SIMPLEXDESCENT, ST_DUALSTEP],
    ),
    (
        "blended_conditional_gradient with hessian, accelerated",
        f,
        cb -> FrankWolfe.blended_conditional_gradient(
            f,
            grad!,
            lmo,
            x0;
            hessian=hessian,
            accelerated=true,
            callback=cb,
            KW...,
        ),
        [ST_REGULAR, ST_SIMPLEXDESCENT, ST_DUALSTEP],
    ),
    (
        "decomposition_invariant_conditional_gradient",
        f,
        cb -> FrankWolfe.decomposition_invariant_conditional_gradient(
            f,
            grad!,
            FrankWolfe.UnitSimplexLMO(1.0),
            collect(x0);
            callback=cb,
            KW...,
        ),
        [ST_PAIRWISE],
    ),
    (
        "blended_decomposition_invariant_conditional_gradient",
        f,
        cb -> FrankWolfe.blended_decomposition_invariant_conditional_gradient(
            f,
            grad!,
            FrankWolfe.UnitSimplexLMO(1.0),
            collect(x0);
            callback=cb,
            KW...,
        ),
        [ST_REGULAR, ST_PAIRWISE],
    ),
    (
        "stochastic_frank_wolfe",
        f_stoch,
        cb -> FrankWolfe.stochastic_frank_wolfe(
            f_stoch,
            lmo,
            collect(x0);
            rng=StableRNG(7),
            batch_size=10,
            max_iteration=200,
            callback=cb,
        ),
        [ST_REGULAR],
    ),
    (
        "corrective AwayStep",
        f,
        corrective(() -> FrankWolfe.AwayStep(false)),
        [ST_REGULAR, ST_AWAY, ST_DROP],
    ),
    (
        "corrective AwayStep lazy",
        f,
        corrective(() -> FrankWolfe.AwayStep(true)),
        [ST_REGULAR, ST_LAZY, ST_AWAY, ST_DROP, ST_DUALSTEP],
    ),
    (
        "corrective BlendedPairwiseStep",
        f,
        corrective(() -> FrankWolfe.BlendedPairwiseStep(false)),
        [ST_REGULAR, ST_PAIRWISE, ST_DROP],
    ),
    (
        "corrective BlendedPairwiseStep lazy",
        f,
        corrective(() -> FrankWolfe.BlendedPairwiseStep(true)),
        [ST_REGULAR, ST_PAIRWISE, ST_DROP, ST_DUALSTEP],
    ),
    (
        "corrective PairwiseStep",
        f,
        corrective(() -> FrankWolfe.PairwiseStep(false)),
        [ST_PAIRWISE, ST_DROP],
    ),
    (
        "corrective PairwiseStep lazy",
        f,
        corrective(() -> FrankWolfe.PairwiseStep(true)),
        [ST_PAIRWISE, ST_DROP, ST_LAZY],
    ),
    (
        "corrective HybridPairAwayStep",
        f,
        corrective(() -> FrankWolfe.HybridPairAwayStep(false, zeros(n))),
        [ST_REGULAR, ST_PAIRWISE, ST_AWAY, ST_DROP],
    ),
    (
        "corrective HybridPairAwayStep lazy",
        f,
        corrective(() -> FrankWolfe.HybridPairAwayStep(true, zeros(n))),
        [ST_REGULAR, ST_PAIRWISE, ST_AWAY, ST_DROP, ST_DUALSTEP],
    ),
    (
        "corrective SimplexGradientDescentStep",
        f,
        corrective(() -> FrankWolfe.SimplexGradientDescentStep(false)),
        [ST_REGULAR, ST_SIMPLEXDESCENT, ST_DROP],
    ),
    (
        "corrective SimplexGradientDescentStep lazy",
        f,
        corrective(() -> FrankWolfe.SimplexGradientDescentStep(true)),
        [ST_REGULAR, ST_SIMPLEXDESCENT, ST_DROP, ST_DUALSTEP],
    ),
    (
        "corrective ProjectedGradientDescentStep",
        f,
        corrective(() -> FrankWolfe.ProjectedGradientDescentStep(hessian=hessian)),
        [ST_REGULAR, ST_SIMPLEXDESCENT],
    ),
    (
        "corrective ProjectedGradientDescentStep lazy",
        f,
        corrective(() -> FrankWolfe.ProjectedGradientDescentStep(hessian=hessian, lazy=true)),
        [ST_REGULAR, ST_SIMPLEXDESCENT, ST_DUALSTEP],
    ),
    (
        "corrective ProjectedGradientDescentStep accelerated",
        f,
        corrective(
            () -> FrankWolfe.ProjectedGradientDescentStep(hessian=hessian, accelerated=true),
        ),
        [ST_REGULAR, ST_SIMPLEXDESCENT],
    ),
]

@testset "Callbacks are called before the step, false cancels it" begin
    @testset "$name" for (name, objective, run, expected_step_types) in SOLVERS
        reference = CallRecord[]
        run(recording_callback(reference, objective))
        step_types = unique(r.step_type for r in reference)
        # the interruptions below cover all interesting step types
        @test expected_step_types ⊆ step_types
        # the first call happens before anything moved away from the starting point
        @test reference[1].x ≈ x0
        # in each state, the gradient was computed at the reported (pre-step) iterate
        @test all(r.grad_ok for r in reference)

        # interrupt at the first call and at the first step of every type
        stops = unique([1; [findfirst(r -> r.step_type == st, reference) for st in step_types]])
        @testset "interrupt at call $k ($(reference[k].step_type))" for k in stops
            records = CallRecord[]
            result = run(recording_callback(records, objective; stop_at=k))
            @test result.status == FrankWolfe.STATUS_INTERRUPTED
            # no step after the interruption
            @test length(records) == k
            @test records[k].step_type == reference[k].step_type
            # the cancelled step was not applied
            @test result.x ≈ records[k].x
        end
    end
end

end # module
