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
# the gradient and the LMO count their calls
const GRAD_CALLS = Ref(0)
function grad!(storage, x)
    GRAD_CALLS[] += 1
    @. storage = 2 * D * (x - xp)
    return nothing
end
const hessian = Matrix(Diagonal(2 .* D))
const lmo = FrankWolfe.TrackingLMO(FrankWolfe.ProbabilitySimplexLMO(1.0))
oracle_calls() = (GRAD_CALLS[], lmo.counter)
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
    v::Union{Nothing,Vector{Float64}}
    d::Union{Nothing,Vector{Float64}}
    gamma::Float64
    dual_gap::Float64
    # Frank-Wolfe gap of `state.gradient` at `state.x`, NaN if not computed
    fw_gap::Float64
    grad_ok::Bool
    has_active_set::Bool
end

is_final(r::CallRecord) = r.step_type in (ST_LAST, ST_POSTPROCESS)

# extreme points of the probability and unit simplices
is_vertex(y) =
    all(yi -> isapprox(yi, 0, atol=1e-12) || isapprox(yi, 1, atol=1e-12), y) && sum(y) <= 1 + 1e-12

"""
Callback recording every call and returning `false` at the `stop_at`-th in-loop call.
`objective` is the objective passed to the algorithm. It identifies the states of the
BCG inner solvers over the reduced problem, whose `x` and `d` are in the space of the
active set weights and are mapped back to the original space.
The Frank-Wolfe gap is computed with an extra LMO call if `compute_fw_gap` is set.
"""
function recording_callback(records, objective; stop_at=0, compute_fw_gap=false)
    steps = 0
    return function callback(state, args...)
        function to_original(y)
            state.f === objective && return collect(y)
            z = zeros(n)
            for (λ, atom) in zip(y, args[1].atoms)
                @. z += λ * atom
            end
            return z
        end
        # the gradient was computed at the reported iterate, i.e., before the step
        grad_ok = if state.grad! === nothing
            true
        else
            g = similar(state.gradient)
            state.grad!(g, state.x)
            isapprox(g, state.gradient, rtol=1e-6, atol=1e-10)
        end
        fw_gap = if compute_fw_gap
            v_fw = FrankWolfe.compute_extreme_point(state.lmo, state.gradient)
            dot(state.gradient, state.x) - dot(state.gradient, v_fw)
        else
            NaN
        end
        record = CallRecord(
            state.step_type,
            to_original(state.x),
            state.v === nothing ? nothing : collect(state.v),
            state.d === nothing ? nothing : to_original(state.d),
            state.gamma,
            state.dual_gap,
            fw_gap,
            grad_ok,
            !isempty(args) && args[1] isa FrankWolfe.AbstractActiveSet,
        )
        push!(records, record)
        if is_final(record)
            return true
        end
        steps += 1
        return steps != stop_at
    end
end

"""
Tests that the final calls happen once after the main loop, with `ST_POSTPROCESS` only on
active-set-based methods, that they describe no step, and that the last one reports the result.
"""
function test_final_calls(records, result)
    final_types = [r.step_type for r in records if is_final(r)]
    @test final_types == (records[1].has_active_set ? [ST_LAST, ST_POSTPROCESS] : [ST_LAST])
    @test all(is_final, records[(end-length(final_types)+1):end])
    @test all(r.d === nothing && r.gamma == 0 for r in records if is_final(r))
    @test records[end].x ≈ result.x
end

# solvers whose `state.dual_gap` is the Frank-Wolfe gap at `state.x` and not a lazy estimate
exact_dual_gap(name) =
    !any(s -> occursin(s, name), ("lazy", "lazified", "blended_conditional_gradient"))

# `step` is a factory: every run must start from a fresh step object so that the interrupted
# reruns replay the reference run, and some steps keep state across solver calls.
corrective(step) =
    (cb; kwargs...) -> FrankWolfe.corrective_frank_wolfe(
        f,
        grad!,
        lmo,
        step(),
        FrankWolfe.ActiveSet([(1.0, x0)]);
        callback=cb,
        KW...,
        kwargs...,
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
        (cb; kwargs...) ->
            FrankWolfe.away_frank_wolfe(f, grad!, lmo, x0; callback=cb, KW..., kwargs...),
        [ST_REGULAR, ST_AWAY, ST_DROP],
    ),
    (
        "away_frank_wolfe lazy",
        f,
        (cb; kwargs...) -> FrankWolfe.away_frank_wolfe(
            f,
            grad!,
            lmo,
            x0;
            lazy=true,
            callback=cb,
            KW...,
            kwargs...,
        ),
        [ST_REGULAR, ST_LAZY, ST_AWAY, ST_DROP, ST_DUALSTEP],
    ),
    (
        "away_frank_wolfe without away steps",
        f,
        (cb; kwargs...) -> FrankWolfe.away_frank_wolfe(
            f,
            grad!,
            lmo,
            x0;
            away_steps=false,
            callback=cb,
            KW...,
            kwargs...,
        ),
        [ST_REGULAR],
    ),
    (
        "pairwise_frank_wolfe",
        f,
        (cb; kwargs...) ->
            FrankWolfe.pairwise_frank_wolfe(f, grad!, lmo, x0; callback=cb, KW..., kwargs...),
        [ST_PAIRWISE, ST_DUALSTEP],
    ),
    (
        "pairwise_frank_wolfe lazy",
        f,
        (cb; kwargs...) -> FrankWolfe.pairwise_frank_wolfe(
            f,
            grad!,
            lmo,
            x0;
            lazy=true,
            callback=cb,
            KW...,
            kwargs...,
        ),
        [ST_PAIRWISE, ST_LAZY, ST_DUALSTEP],
    ),
    (
        "blended_pairwise_conditional_gradient",
        f,
        (cb; kwargs...) -> FrankWolfe.blended_pairwise_conditional_gradient(
            f,
            grad!,
            lmo,
            x0;
            callback=cb,
            KW...,
            kwargs...,
        ),
        [ST_REGULAR, ST_PAIRWISE, ST_DROP],
    ),
    (
        "blended_pairwise_conditional_gradient lazy",
        f,
        (cb; kwargs...) -> FrankWolfe.blended_pairwise_conditional_gradient(
            f,
            grad!,
            lmo,
            x0;
            lazy=true,
            callback=cb,
            KW...,
            kwargs...,
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
        "decomposition_invariant_conditional_gradient lazy",
        f,
        cb -> FrankWolfe.decomposition_invariant_conditional_gradient(
            f,
            grad!,
            FrankWolfe.UnitSimplexLMO(1.0),
            collect(x0);
            lazy=true,
            callback=cb,
            KW...,
        ),
        [ST_PAIRWISE, ST_LAZY, ST_DUALSTEP],
    ),
    (
        "blended_decomposition_invariant_conditional_gradient lazy",
        f,
        cb -> FrankWolfe.blended_decomposition_invariant_conditional_gradient(
            f,
            grad!,
            FrankWolfe.UnitSimplexLMO(1.0),
            collect(x0);
            lazy=true,
            callback=cb,
            KW...,
        ),
        [ST_REGULAR, ST_PAIRWISE, ST_DUALSTEP],
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
        result = run(recording_callback(reference, objective; compute_fw_gap=exact_dual_gap(name)))
        test_final_calls(reference, result)
        steps = filter(!is_final, reference)
        step_types = unique(r.step_type for r in steps)
        # the interruptions below cover all interesting step types
        @test expected_step_types ⊆ step_types
        # the first call happens before anything moved away from the starting point
        @test steps[1].x ≈ x0
        # in each state, the gradient was computed at the reported (pre-step) iterate
        @test all(r.grad_ok for r in reference)
        # the dual gap is computed at the reported iterate
        if exact_dual_gap(name)
            @test all(isapprox(r.dual_gap, r.fw_gap, rtol=1e-6, atol=1e-8) for r in steps)
        end
        # the vertex is an extreme point and defines the direction: d = x - v (FW step),
        # d = v - x (away step), or d = a - v with an extreme point a (pairwise step)
        @test all(r.v === nothing || is_vertex(r.v) for r in reference)
        @test all(steps) do r
            return r.v === nothing ||
                   r.d === nothing ||
                   isapprox(r.d, r.x - r.v, atol=1e-10) ||
                   isapprox(r.d, r.v - r.x, atol=1e-10) ||
                   is_vertex(r.d + r.v)
        end
        # each step moves from x to x - gamma * d, the following call reports that iterate
        @test all(eachindex(steps)) do i
            r = steps[i]
            x_next = r.d === nothing ? r.x : r.x - r.gamma * r.d
            return isapprox(reference[i+1].x, x_next, rtol=1e-6, atol=1e-8)
        end
        # steps that do not move the iterate have no direction and a zero step size
        @test all(r.gamma == 0 for r in steps if r.d === nothing)
        # dual steps only update the gap estimate, except in the lazified CG where they are
        # Frank-Wolfe steps towards the vertex computed by the LMO
        if name != "lazified_conditional_gradient"
            @test all(r.d === nothing for r in steps if r.step_type == ST_DUALSTEP)
        end

        # interrupt at the first call and at the first step of every type
        stops = unique([1; [findfirst(r -> r.step_type == st, steps) for st in step_types]])
        @testset "interrupt at call $k ($(steps[k].step_type))" for k in stops
            records = CallRecord[]
            result = run(recording_callback(records, objective; stop_at=k))
            @test result.status == FrankWolfe.STATUS_INTERRUPTED
            # no step after the interruption
            interrupted_steps = filter(!is_final, records)
            @test length(interrupted_steps) == k
            @test interrupted_steps[k].step_type == steps[k].step_type
            # the cancelled step was not applied
            @test result.x ≈ interrupted_steps[k].x
            test_final_calls(records, result)
        end
    end
end

# the variants with the `recompute_last_vertex` keyword argument
has_recompute_last_vertex(name) = any(
    prefix -> startswith(name, prefix),
    (
        "away_frank_wolfe",
        "pairwise_frank_wolfe",
        "blended_pairwise_conditional_gradient",
        "corrective",
    ),
)

"""
Runs the solver and returns the result, the step types passed to the callback, and the
numbers of gradient and LMO calls.
"""
function run_counting(run; with_callback=true, kwargs...)
    step_types = FrankWolfe.StepType[]
    callback = (state, args...) -> (push!(step_types, state.step_type); true)
    calls_before = oracle_calls()
    result = run(with_callback ? callback : nothing; kwargs...)
    return result, step_types, oracle_calls() .- calls_before
end

@testset "recompute_last_vertex = false skips everything after the main loop" begin
    @testset "$name" for (name, _, run, _) in filter(s -> has_recompute_last_vertex(s[1]), SOLVERS)
        reference, reference_step_types, reference_calls = run_counting(run)
        @test reference_step_types[(end-1):end] == [ST_LAST, ST_POSTPROCESS]

        result, step_types, calls = run_counting(run; recompute_last_vertex=false)
        # no final callbacks, the calls in the main loop are the same
        @test step_types == reference_step_types[1:(end-2)]
        # no gradient and LMO calls for ST_LAST and ST_POSTPROCESS
        @test reference_calls == calls .+ (2, 2)
        # the returned values are the ones of the main loop
        @test result.status == reference.status
        @test result.x ≈ reference.x
        @test result.primal ≈ reference.primal
        @test is_vertex(result.v)
        @test result.dual_gap >= 0
    end
end

end # module
