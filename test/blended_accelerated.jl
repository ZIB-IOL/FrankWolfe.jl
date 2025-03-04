using LinearAlgebra
using Random
using SparseArrays
using Test
using Test
using FrankWolfe
using DoubleFloats
using StableRNGs

n = 200
k = 200

s = 42
Random.seed!(StableRNG(s), s)

const matrix = rand(n, n)
const hessian = transpose(matrix) * matrix
const linear = rand(n)
f(x) = dot(linear, x) + 0.5 * transpose(x) * hessian * x
function grad!(storage, x)
    return storage .= linear + hessian * x
end
const L = eigmax(hessian)

# This test covers the generic types with accelerated BCG
# only few iterations are run because linear algebra with BigFloat is intensive
@testset "Type $T" for T in (Float64, Double64, BigFloat)
    @testset "LMO $(typeof(lmo)) Probability simplex" for lmo in (
        FrankWolfe.ProbabilitySimplexOracle{T}(1.0),
        FrankWolfe.KSparseLMO{T}(100, 100.0),
    )
        x0 = FrankWolfe.compute_extreme_point(lmo, spzeros(T, n))

        target_tolerance = 1e-5

        x, v, primal_accel, dual_gap_accel, _, _ = FrankWolfe.blended_conditional_gradient(
            f,
            grad!,
            lmo,
            copy(x0),
            epsilon=target_tolerance,
            max_iteration=k,
            line_search=FrankWolfe.AdaptiveZerothOrder(L_est=L),
            print_iter=k / 10,
            hessian=hessian,
            memory_mode=FrankWolfe.InplaceEmphasis(),
            accelerated=true,
            verbose=false,
            trajectory=false,
            lazy_tolerance=1.0,
            weight_purge_threshold=1e-10,
        )

        x, v, primal_hessian, dual_gap_hessian, _, _ = FrankWolfe.blended_conditional_gradient(
            f,
            grad!,
            lmo,
            copy(x0),
            epsilon=target_tolerance,
            max_iteration=k,
            line_search=FrankWolfe.AdaptiveZerothOrder(L_est=L),
            print_iter=k / 10,
            hessian=hessian,
            memory_mode=FrankWolfe.InplaceEmphasis(),
            accelerated=false,
            verbose=false,
            trajectory=false,
            lazy_tolerance=1.0,
            weight_purge_threshold=1e-10,
        )

        x, v, primal_bcg, dual_gap_bcg, _, _ = FrankWolfe.blended_conditional_gradient(
            f,
            grad!,
            lmo,
            copy(x0),
            epsilon=target_tolerance,
            max_iteration=k,
            line_search=FrankWolfe.AdaptiveZerothOrder(L_est=L),
            print_iter=k / 10,
            memory_mode=FrankWolfe.InplaceEmphasis(),
            verbose=false,
            trajectory=false,
            lazy_tolerance=1.0,
            weight_purge_threshold=1e-10,
        )
    end
end
