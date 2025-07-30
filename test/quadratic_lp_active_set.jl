using FrankWolfe
using LinearAlgebra
using Random
using Test
import HiGHS
import MathOptInterface as MOI
using StableRNGs

n = Int(1e4)
k = 5000

s = 10
rng = StableRNG(s)
Random.seed!(rng, s)

xpi = rand(rng, n);
total = sum(xpi);

const xp = xpi ./ total;

f(x) = norm(x - xp)^2
function grad!(storage, x)
    @. storage = 2 * (x - xp)
end

lmo = FrankWolfe.KSparseLMO(5, 1.0)

const x00 = FrankWolfe.compute_extreme_point(lmo, rand(rng, n))

function build_callback(trajectory_arr)
    return function callback(state, active_set, args...)
        return push!(trajectory_arr, (FrankWolfe.callback_state(state)..., length(active_set)))
    end
end

trajectoryBPCG_standard = []
x, v, primal, dual_gap, _ = FrankWolfe.blended_pairwise_conditional_gradient(
    f,
    grad!,
    lmo,
    copy(x00),
    max_iteration=k,
    line_search=FrankWolfe.Shortstep(2.0),
    verbose=false,
    callback=build_callback(trajectoryBPCG_standard),
);

as_quad_direct = FrankWolfe.ActiveSetQuadraticLinearSolve(
    [(1.0, copy(x00))],
    2 * LinearAlgebra.I,
    -2xp,
    MOI.instantiate(MOI.OptimizerWithAttributes(HiGHS.Optimizer, MOI.Silent() => true)),
)

trajectoryBPCG_quadratic_direct_specialized = []
x, v, primal, dual_gap, _ = FrankWolfe.blended_pairwise_conditional_gradient(
    f,
    grad!,
    lmo,
    as_quad_direct,
    max_iteration=k,
    line_search=FrankWolfe.Shortstep(2.0),
    verbose=false,
    callback=build_callback(trajectoryBPCG_quadratic_direct_specialized),
);

as_quad_direct_generic = FrankWolfe.ActiveSetQuadraticLinearSolve(
    [(1.0, copy(x00))],
    2 * Diagonal(ones(length(xp))),
    -2xp,
    MOI.instantiate(MOI.OptimizerWithAttributes(HiGHS.Optimizer, MOI.Silent() => true)),
)

trajectoryBPCG_quadratic_direct_generic = []
x, v, primal, dual_gap, _ = FrankWolfe.blended_pairwise_conditional_gradient(
    f,
    grad!,
    lmo,
    as_quad_direct_generic,
    max_iteration=k,
    line_search=FrankWolfe.Shortstep(2.0),
    verbose=false,
    callback=build_callback(trajectoryBPCG_quadratic_direct_generic),
);

as_quad_direct_basic_as = FrankWolfe.ActiveSetQuadraticLinearSolve(
    FrankWolfe.ActiveSet([1.0], [copy(x00)], collect(x00)),
    2 * LinearAlgebra.I,
    -2xp,
    MOI.instantiate(MOI.OptimizerWithAttributes(HiGHS.Optimizer, MOI.Silent() => true)),
)

trajectoryBPCG_quadratic_noqas = []
x, v, primal, dual_gap, _ = FrankWolfe.blended_pairwise_conditional_gradient(
    f,
    grad!,
    lmo,
    as_quad_direct_basic_as,
    max_iteration=k,
    line_search=FrankWolfe.Shortstep(2.0),
    verbose=false,
    callback=build_callback(trajectoryBPCG_quadratic_noqas),
);

as_quad_direct_product_caching = FrankWolfe.ActiveSetQuadraticLinearSolve(
    FrankWolfe.ActiveSetQuadraticProductCaching([(1.0, copy(x00))], 2 * LinearAlgebra.I, -2xp),
    2 * LinearAlgebra.I,
    -2xp,
    MOI.instantiate(MOI.OptimizerWithAttributes(HiGHS.Optimizer, MOI.Silent() => true)),
)

trajectoryBPCG_quadratic_direct_product_caching = []
x, v, primal, dual_gap, _ = FrankWolfe.blended_pairwise_conditional_gradient(
    f,
    grad!,
    lmo,
    as_quad_direct_product_caching,
    max_iteration=k,
    line_search=FrankWolfe.Shortstep(2.0),
    verbose=false,
    callback=build_callback(trajectoryBPCG_quadratic_direct_product_caching),
);

dual_gaps_quadratic_specialized = getindex.(trajectoryBPCG_quadratic_direct_specialized, 4)
dual_gaps_quadratic_generic = getindex.(trajectoryBPCG_quadratic_direct_generic, 4)
dual_gaps_quadratic_noqas = getindex.(trajectoryBPCG_quadratic_noqas, 4)
dual_gaps_bpcg = getindex.(trajectoryBPCG_standard, 4)
dual_gaps_quadratic_direct_product_caching =
    getindex.(trajectoryBPCG_quadratic_direct_product_caching, 4)


@test dual_gaps_quadratic_specialized[end] < dual_gaps_bpcg[end]
@test dual_gaps_quadratic_noqas[end] < dual_gaps_bpcg[end]
@test dual_gaps_quadratic_direct_product_caching[end] < dual_gaps_bpcg[end]
@test norm(dual_gaps_quadratic_noqas - dual_gaps_quadratic_noqas) ≤ k * 1e-5
