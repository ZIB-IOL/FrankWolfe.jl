module Test_blended_accelerated

using LinearAlgebra
using Random
using SparseArrays
using Test
using FrankWolfe
using DoubleFloats
using StableRNGs

n = 200
k = 200

s = 42
rng = StableRNG(s)
Random.seed!(rng, s)

const matrix = rand(rng, n, n)
const hessian = transpose(matrix) * matrix
const linear = rand(rng, n)
const L = eigmax(hessian)

function quadratic(linear_term, hessian_term)
    f(x) = dot(linear_term, x) + 0.5 * transpose(x) * hessian_term * x
    grad!(storage, x) = (storage .= linear_term + hessian_term * x)
    return f, grad!
end

# The degenerate objective appends a coordinate without curvature. Its cost is below the
# optimal gradient level (≈ 43) over the probability simplex, so its vertex enters the active
# set, and from then on every reduced Hessian is singular (mu = 0).
const objectives = (
    "nondegenerate" => (linear, hessian),
    "degenerate" => ([linear; 30.0], [hessian zeros(n); zeros(1, n + 1)]),
)

# This test covers the generic types with (accelerated) BCG.
# Over the K-sparse polytope, BCG converges slowly, and only a few iterations are run
# because linear algebra with BigFloat is intensive.
@testset "Type $T" for T in (Float64, Double64, BigFloat)
    @testset "$name objective" for (name, (linear_term, hessian_term)) in objectives
        f, grad! = quadratic(linear_term, hessian_term)
        dim = length(linear_term)
        @testset "LMO $(typeof(lmo))" for (lmo, max_iteration, converges) in (
            (FrankWolfe.ProbabilitySimplexLMO{T}(1.0), 2000, true),
            (FrankWolfe.KSparseLMO{T}(100, 100.0), k, false),
        )
            x0 = FrankWolfe.compute_extreme_point(lmo, spzeros(T, dim))
            target_tolerance = 1e-5
            # accelerated, with Hessian, without Hessian
            variants = ((hessian_term, true), (hessian_term, false), (nothing, false))
            results = map(variants) do (hessian_bcg, accelerated)
                return FrankWolfe.blended_conditional_gradient(
                    f,
                    grad!,
                    lmo,
                    copy(x0),
                    epsilon=target_tolerance,
                    max_iteration=max_iteration,
                    line_search=FrankWolfe.AdaptiveZerothOrder(L_est=L),
                    print_iter=max_iteration / 10,
                    hessian=hessian_bcg,
                    accelerated=accelerated,
                    memory_mode=FrankWolfe.InplaceEmphasis(),
                    verbose=false,
                    trajectory=false,
                    sparsity_control=1.0,
                    weight_purge_threshold=1e-10,
                )
            end
            if converges
                for result in results
                    @test result.status == FrankWolfe.STATUS_OPTIMAL
                    @test result.primal ≈ results[end].primal atol = target_tolerance
                end
            end
        end
    end
end

end # module
